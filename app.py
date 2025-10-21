"""
DC Retail Case & Bin Label Generator
===================================

A Streamlit application for processing CSV files and generating Zebra printer labels.
Supports direct printing to Zebra ZD410/ZD621 printers via network connection and Browser Print.
Handles partial case/bin quantities correctly.

Author: DC Retail
Version: 2.3.0

VERSION HISTORY:
- v2.3.0 (2025-10-20): Added Zebra Browser Print integration for cloud printing
- v2.2.1 (2025-09-29): Fixed product name wrapping on large labels, added version display
- v2.2.0 (2025-09-29): Added 4" × 2" label size with proportional scaling
- v2.1.0 (2025-09-29): Hybrid positioning (batch follows product, rest fixed), Z-A sorting
- v2.0.0 (2025-09-27): Complete rewrite with partial quantity handling, individual items support
- v1.0.0 (2025): Initial release
"""

import streamlit as st
import pandas as pd
import io
import math
import socket
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import streamlit.components.v1 as components
import json
import base64

# Version
VERSION = "2.3.0"

# Import QR code libraries with error handling
try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="DC Retail Case & Bin Label Generator",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ DC Retail Case & Bin Label Generator")
st.markdown("**DC Retail** | Process CSV files for label printing with calculated quantities")

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    session_vars = ['processed_data', 'sales_order_data', 'products_data', 'packages_data', 
                    'browser_print_available', 'selected_printer']
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

initialize_session_state()

# =============================================================================
# BROWSER PRINT INTEGRATION
# =============================================================================

def generate_browser_print_script() -> str:
    """Generate the Browser Print detection and setup script"""
    return """
    <script type="text/javascript">
    // Browser Print detection and initialization
    window.BrowserPrintAvailable = false;
    window.SelectedPrinter = null;
    window.AvailablePrinters = [];
    
    // Load Browser Print script dynamically
    function loadBrowserPrint() {
        // First check if Browser Print is running by trying to access it
        fetch('http://localhost:9100/available', {mode: 'no-cors'})
            .then(function() {
                // If we get here, Browser Print is likely running
                loadBrowserPrintScript();
            })
            .catch(function() {
                console.log('Browser Print not detected');
                window.parent.postMessage({
                    type: 'browserPrintStatus',
                    available: false,
                    message: 'Browser Print not detected. Please install and run Zebra Browser Print.'
                }, '*');
            });
    }
    
    function loadBrowserPrintScript() {
        var script = document.createElement('script');
        script.src = 'http://localhost:9100/JSPrintClient.js';
        script.onload = function() {
            initializeBrowserPrint();
        };
        script.onerror = function() {
            console.error('Failed to load Browser Print script');
            window.parent.postMessage({
                type: 'browserPrintStatus',
                available: false,
                message: 'Failed to load Browser Print. Ensure it is running.'
            }, '*');
        };
        document.head.appendChild(script);
    }
    
    function initializeBrowserPrint() {
        if (typeof BrowserPrint !== 'undefined') {
            window.BrowserPrintAvailable = true;
            
            // Get available printers
            BrowserPrint.getDefaultDevice('printer', function(device) {
                if (device && device.name) {
                    window.SelectedPrinter = device;
                    
                    // Get all local printers
                    BrowserPrint.getLocalDevices(function(printers) {
                        window.AvailablePrinters = printers;
                        
                        // Send status to Streamlit
                        window.parent.postMessage({
                            type: 'browserPrintStatus',
                            available: true,
                            defaultPrinter: device.name,
                            printers: printers.map(p => p.name)
                        }, '*');
                    }, function() {
                        window.parent.postMessage({
                            type: 'browserPrintStatus',
                            available: true,
                            defaultPrinter: device.name,
                            printers: [device.name]
                        }, '*');
                    });
                }
            }, function(error) {
                console.error('Browser Print error:', error);
                window.parent.postMessage({
                    type: 'browserPrintStatus',
                    available: false,
                    message: 'Browser Print detected but no printers found.'
                }, '*');
            });
        }
    }
    
    // Function to print ZPL
    window.printZPL = function(zplData, printerName) {
        if (!window.BrowserPrintAvailable) {
            alert('Browser Print is not available. Please install Zebra Browser Print.');
            return;
        }
        
        // Find the printer
        var selectedPrinter = window.SelectedPrinter;
        if (printerName && window.AvailablePrinters.length > 0) {
            var found = window.AvailablePrinters.find(p => p.name === printerName);
            if (found) selectedPrinter = found;
        }
        
        if (!selectedPrinter) {
            alert('No printer selected or available.');
            return;
        }
        
        // Send ZPL to printer
        selectedPrinter.send(zplData, 
            function() {
                window.parent.postMessage({
                    type: 'printComplete',
                    success: true,
                    message: 'Labels sent to printer successfully!'
                }, '*');
            },
            function(error) {
                console.error('Print error:', error);
                window.parent.postMessage({
                    type: 'printComplete',
                    success: false,
                    message: 'Print failed: ' + error
                }, '*');
            }
        );
    };
    
    // Initialize on load
    window.addEventListener('load', loadBrowserPrint);
    
    // Listen for print commands from Streamlit
    window.addEventListener('message', function(event) {
        if (event.data.type === 'printZPL') {
            window.printZPL(event.data.zplData, event.data.printerName);
        }
    });
    </script>
    """

def inject_browser_print_detector():
    """Inject the Browser Print detection script into the page"""
    components.html(generate_browser_print_script(), height=0)

def send_to_browser_print(zpl_data: str, printer_name: Optional[str] = None) -> None:
    """Send ZPL data to Browser Print via JavaScript injection"""
    
    # Escape the ZPL data for JavaScript
    escaped_zpl = zpl_data.replace('\\', '\\\\').replace('`', '\\`').replace('"', '\\"')
    
    print_script = f"""
    <script>
    // Send print command to parent window
    window.parent.postMessage({{
        type: 'printZPL',
        zplData: `{escaped_zpl}`,
        printerName: {'null' if not printer_name else f'"{printer_name}"'}
    }}, '*');
    </script>
    """
    
    components.html(print_script, height=0)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_numeric(value, default=0):
    """Convert any value to numeric, handling strings, NaN, None, etc."""
    if pd.isna(value) or value is None or value == '':
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                return default
        num_val = float(value)
        return int(num_val) if num_val.is_integer() else num_val
    except (ValueError, TypeError, AttributeError):
        return default

def safe_sum(series):
    """Safely sum a series that might contain strings"""
    return sum(safe_numeric(x) for x in series)

def safe_count_nonzero(series):
    """Count non-zero values in a series that might contain strings"""
    return sum(1 for x in series if safe_numeric(x) > 0)

def calculate_individual_quantities(package_qty: float, container_qty: float) -> List[float]:
    """
    Calculate individual label quantities, handling partials correctly.
    Special case: When container_qty = 1, treat as individual items (1 label with full qty).
    """
    if package_qty <= 0 or container_qty <= 0:
        return []
    
    if container_qty == 1:
        return [package_qty]
    
    quantities = []
    remaining = package_qty
    
    while remaining > 0:
        if remaining >= container_qty:
            quantities.append(container_qty)
            remaining -= container_qty
        else:
            quantities.append(remaining)
            remaining = 0
    
    return quantities

# =============================================================================
# FILE LOADING
# =============================================================================

def load_sales_order_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """Load Sales Order CSV with special handling for metadata lines."""
    try:
        parsing_methods = [
            {'skiprows': 3, 'dtype': str},
            {'skiprows': 3, 'encoding': 'utf-8', 'quotechar': '"', 'skipinitialspace': True, 'dtype': str},
            {'skiprows': 3, 'sep': ';', 'encoding': 'utf-8', 'dtype': str},
            {'skiprows': 3, 'sep': '\t', 'encoding': 'utf-8', 'dtype': str}
        ]
        
        for method in parsing_methods:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, **method)
                if not df.empty:
                    return df
            except Exception:
                continue
                
        return None
    except Exception as e:
        st.error(f"Error loading Sales Order CSV: {str(e)}")
        return None

def load_standard_csv(uploaded_file, file_type: str) -> Optional[pd.DataFrame]:
    """Load standard CSV files with string preservation"""
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error loading {file_type} CSV: {str(e)}")
        return None

def format_delivery_date(date_series: pd.Series) -> pd.Series:
    """Format delivery dates to mm-dd-yy format"""
    try:
        date_series = pd.to_datetime(date_series, errors='coerce')
        return date_series.dt.strftime('%m-%d-%y')
    except Exception:
        return date_series.astype(str)

# =============================================================================
# DATA PROCESSING
# =============================================================================

def calculate_labels_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate how many case and bin labels are needed for each row."""
    if 'Package Quantity' in df.columns and 'Case Quantity' in df.columns:
        case_labels = []
        for _, row in df.iterrows():
            pkg_qty = safe_numeric(row.get('Package Quantity', 0))
            case_qty = safe_numeric(row.get('Case Quantity', 0))
            
            if pkg_qty > 0 and case_qty > 0:
                if case_qty == 1:
                    case_labels.append(1)
                else:
                    case_labels.append(math.ceil(pkg_qty / case_qty))
            else:
                case_labels.append(0)
        df['Case Labels Needed'] = case_labels
    else:
        df['Case Labels Needed'] = 0
    
    if 'Package Quantity' in df.columns and 'Bin Quantity' in df.columns:
        bin_labels = []
        for _, row in df.iterrows():
            pkg_qty = safe_numeric(row.get('Package Quantity', 0))
            bin_qty = safe_numeric(row.get('Bin Quantity', 0))
            
            if pkg_qty > 0 and bin_qty > 0:
                if bin_qty == 1:
                    bin_labels.append(1)
                else:
                    bin_labels.append(math.ceil(pkg_qty / bin_qty))
            else:
                bin_labels.append(0)
        df['Bin Labels Needed'] = bin_labels
    else:
        df['Bin Labels Needed'] = 0
    
    return df

def merge_data_sources(sales_order_df: pd.DataFrame, 
                      products_df: pd.DataFrame, 
                      packages_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merge the three data sources and calculate label requirements."""
    try:
        st.session_state.sales_order_data = sales_order_df
        st.session_state.products_data = products_df
        st.session_state.packages_data = packages_df
        
        # Find the ID column (might be 'ID', 'Id', or 'Product ID')
        products_id_col = None
        for col in ['ID', 'Id', 'Product ID', 'Product Id', 'id']:
            if col in products_df.columns:
                products_id_col = col
                break
        
        if not products_id_col:
            st.error(f"Cannot find ID column in Products CSV. Available columns: {products_df.columns.tolist()}")
            return None
        
        merged_df = sales_order_df.merge(
            products_df, 
            left_on='Product Id', 
            right_on=products_id_col, 
            how='left',
            suffixes=('', '_products')
        )
        
        final_df = merged_df.merge(
            packages_df, 
            left_on='Package Label', 
            right_on='Package Label', 
            how='left',
            suffixes=('', '_packages')
        )
        
        column_mapping = {
            'Product': 'Product Name',
            'Category': 'Category',
            'Package Batch Number': 'Batch No',
            'Package Label': 'Package Label',
            'Quantity': 'Package Quantity',
            'Units Per Case': 'Case Quantity',
            'Bin Quantity (Retail)': 'Bin Quantity',
            'Customer': 'Customer',
            'Invoice Numbers': 'Invoice No',
            'Metrc Manifest Number': 'METRC Manifest',
            'Delivery Date': 'Delivery Date',
            'Order Number': 'Sales Order Number',
            'Sell By': 'Sell by'
        }
        
        clean_data = pd.DataFrame()
        for original_col, new_col in column_mapping.items():
            if original_col in final_df.columns:
                clean_data[new_col] = final_df[original_col]
            else:
                clean_data[new_col] = None
        
        if 'Delivery Date' in clean_data.columns:
            clean_data['Delivery Date'] = format_delivery_date(clean_data['Delivery Date'])
        
        clean_data = calculate_labels_needed(clean_data)
        
        desired_order = [
            'Product Name', 'Category', 'Batch No', 'Package Label', 
            'Package Quantity', 'Case Quantity', 'Bin Quantity', 
            'Case Labels Needed', 'Bin Labels Needed',
            'Customer', 'Invoice No', 'Sales Order Number', 'METRC Manifest', 
            'Delivery Date', 'Sell by'
        ]
        
        for col in desired_order:
            if col not in clean_data.columns:
                clean_data[col] = None
        
        return clean_data[desired_order]
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# =============================================================================
# LABEL GENERATION
# =============================================================================

def sanitize_qr_data(package_label) -> str:
    """Preserve complete package label for QR code."""
    if pd.isna(package_label) or package_label is None:
        return ""
    return str(package_label).strip()

def generate_label_zpl(product_name: str, batch_no: str, qty: float,
                      pkg_qty: str, date_str: str, package_label: str, 
                      sell_by: str, invoice_no: str, metrc_manifest: str, category: str,
                      label_type: str = "Case",
                      label_width: float = 1.75, label_height: float = 0.875, 
                      dpi: int = 300) -> str:
    """Generate ZPL code for Zebra printer labels with proportional scaling."""
    width_dots = int(label_width * dpi)
    height_dots = int(label_height * dpi)
    
    # Calculate scaling factor (reference: 1.75" × 0.875" at 300dpi)
    reference_width = 1.75 * 300
    reference_height = 0.875 * 300
    scale_x = width_dots / reference_width
    scale_y = height_dots / reference_height
    scale = (scale_x + scale_y) / 2
    
    # Font sizes - scaled
    fonts = {
        'extra_large': int(32 * scale),
        'large': int(28 * scale),
        'medium': int(20 * scale),
        'large_plus': int(24 * scale),
        'small': int(16 * scale),
        'small_plus': int(18 * scale)
    }
    
    # Layout constants - scaled
    left_margin = int(30 * scale)
    line_spacing = int(35 * scale)
    
    # Fixed positions - scaled
    quantity_y = int(120 * scale)
    category_y = int(160 * scale)
    delivered_y = int(195 * scale)
    sell_by_y = int(220 * scale)
    
    qr_data = sanitize_qr_data(package_label)
    
    # Handle product name wrapping - keep character limit CONSTANT regardless of scale
    # Larger fonts with bigger labels still need same character limits to fit properly
    product_name = str(product_name) if pd.notna(product_name) else ""
    product_lines = []
    char_limit = 35  # Keep constant - don't scale this
    
    if len(product_name) > char_limit:
        words = product_name.split()
        line1 = ""
        line2 = ""
        
        for word in words:
            if len(line1 + " " + word) <= char_limit and not line2:
                line1 = (line1 + " " + word).strip()
            else:
                line2 = (line2 + " " + word).strip()
        
        if len(line2) > char_limit:
            line2 = line2[:32] + "..."
            
        product_lines = [line1, line2] if line2 else [line1]
    else:
        product_lines = [product_name]
    
    # Format dates
    try:
        if pd.notna(date_str) and date_str and str(date_str).strip():
            date_obj = pd.to_datetime(date_str)
            formatted_date = date_obj.strftime('%m/%d/%Y')
        else:
            formatted_date = datetime.now().strftime('%m/%d/%Y')
    except Exception:
        formatted_date = str(date_str) if date_str else datetime.now().strftime('%m/%d/%Y')
    
    sell_by_formatted = ""
    if pd.notna(sell_by) and sell_by and str(sell_by).strip():
        try:
            sell_by_obj = pd.to_datetime(sell_by)
            sell_by_formatted = sell_by_obj.strftime('%m/%d/%Y')
        except Exception:
            sell_by_formatted = str(sell_by)
    
    # QR code - scaled
    qr_size = max(5, int(5 * scale))
    qr_x = width_dots - int(140 * scale)
    qr_y = height_dots - int(160 * scale)
    
    # Combined bottom line
    combined_parts = []
    if invoice_no:
        combined_parts.append(str(invoice_no))
    if metrc_manifest:
        combined_parts.append(str(metrc_manifest))
    if qr_data:
        display_limit = int(25 * scale)
        display_package = qr_data[:display_limit] + "..." if len(qr_data) > display_limit else qr_data
        combined_parts.append(display_package)
    
    combined_text = " | ".join(combined_parts)
    
    # Build ZPL
    zpl_lines = ["^XA", f"^CF0,{fonts['large']}"]
    
    # Product name - dynamic
    current_y = int(20 * scale)
    for line in product_lines:
        zpl_lines.append(f"^FO{left_margin},{current_y}^FD{line}^FS")
        current_y += line_spacing
    
    # Batch - dynamic (follows product)
    zpl_lines.extend([
        f"^CF0,{fonts['medium']}",
        f"^FO{left_margin},{current_y}^FDBatch: {batch_no}^FS"
    ])
    
    # Quantity - fixed
    if qty == int(qty):
        qty_display = f"{label_type} Qty: {int(qty)}"
    else:
        qty_display = f"{label_type} Qty: {qty:.1f}"
        
    zpl_lines.extend([
        f"^CF0,{fonts['extra_large']}",
        f"^FO{left_margin},{quantity_y}^FD{qty_display}^FS"
    ])
    
    # Category - fixed
    category_text = str(category) if pd.notna(category) else ""
    if category_text:
        zpl_lines.extend([
            f"^CF0,{fonts['large']}",
            f"^FO{left_margin},{category_y}^FD{category_text}^FS"
        ])
    
    # Delivered - fixed
    zpl_lines.extend([
        f"^CF0,{fonts['large_plus']}",
        f"^FO{left_margin},{delivered_y}^FDDelivered: {formatted_date}^FS"
    ])
    
    # Sell by + Pkg Qty on same line - fixed
    sell_by_line = ""
    if sell_by_formatted:
        sell_by_line = f"Sell By: {sell_by_formatted}"
    
    if pkg_qty:
        if sell_by_line:
            combined_sell_pkg = f"{sell_by_line} | Pkg Qty: {pkg_qty}"
        else:
            combined_sell_pkg = f"Pkg Qty: {pkg_qty}"
        
        zpl_lines.extend([
            f"^CF0,{fonts['large_plus']}",
            f"^FO{left_margin},{sell_by_y}^FD{combined_sell_pkg}^FS"
        ])
    elif sell_by_line:
        zpl_lines.extend([
            f"^CF0,{fonts['large_plus']}",
            f"^FO{left_margin},{sell_by_y}^FD{sell_by_line}^FS"
        ])
    
    # QR code - scaled
    if qr_data:
        zpl_lines.append(f"^FO{qr_x},{qr_y}^BQN,2,{qr_size}^FDQA,{qr_data}^FS")
    
    # Combined bottom line - scaled
    bottom_line_y = height_dots - int(15 * scale)
    if combined_text:
        zpl_lines.extend([
            f"^CF0,{fonts['small_plus']}",
            f"^FO{left_margin},{bottom_line_y}^FD{combined_text}^FS"
        ])
    
    zpl_lines.append("^XZ")
    return "\n".join(zpl_lines)

def generate_all_labels_for_row(row: pd.Series, label_type: str, 
                               label_width: float, label_height: float, dpi: int) -> List[str]:
    """Generate all labels needed for a single row."""
    labels = []
    
    package_qty = safe_numeric(row.get('Package Quantity', 0))
    
    if label_type == "Case":
        container_qty = safe_numeric(row.get('Case Quantity', 0))
        zpl_label_type = "Case"
    else:
        container_qty = safe_numeric(row.get('Bin Quantity', 0))
        zpl_label_type = "Bin"
    
    if package_qty <= 0 or container_qty <= 0:
        return []
    
    individual_quantities = calculate_individual_quantities(package_qty, container_qty)
    
    for qty in individual_quantities:
        zpl = generate_label_zpl(
            product_name=row.get('Product Name', ''),
            batch_no=row.get('Batch No', ''),
            qty=qty,
            pkg_qty=row.get('Package Quantity', ''),
            date_str=row.get('Delivery Date', ''),
            package_label=row.get('Package Label', ''),
            sell_by=row.get('Sell by', ''),
            invoice_no=row.get('Invoice No', ''),
            metrc_manifest=row.get('METRC Manifest', ''),
            category=row.get('Category', ''),
            label_type=zpl_label_type,
            label_width=label_width,
            label_height=label_height,
            dpi=dpi
        )
        labels.append(zpl)
    
    return labels

def generate_labels_for_dataset(df: pd.DataFrame, label_type: str, 
                               label_width: float, label_height: float, dpi: int) -> List[str]:
    """Generate all labels for entire dataset, sorted Z-A by Product Name"""
    all_labels = []
    
    # Sort Z-A (descending)
    df_sorted = df.sort_values('Product Name', ascending=False, na_position='last')
    
    for _, row in df_sorted.iterrows():
        if label_type == "Case" and safe_numeric(row.get('Case Labels Needed', 0)) > 0:
            row_labels = generate_all_labels_for_row(row, "Case", label_width, label_height, dpi)
            if row_labels:
                all_labels.extend(row_labels)
        elif label_type == "Bin" and safe_numeric(row.get('Bin Labels Needed', 0)) > 0:
            row_labels = generate_all_labels_for_row(row, "Bin", label_width, label_height, dpi)
            if row_labels:
                all_labels.extend(row_labels)
    
    return all_labels

def generate_filename(data: pd.DataFrame, label_type: str, label_width: float, 
                     label_height: float, dpi: int) -> str:
    """Generate smart filename based on data content and label specifications."""
    size_spec = f"{label_width}x{label_height}"
    dpi_spec = f"{dpi}dpi"
    
    customers = data['Customer'].dropna().unique()
    if len(customers) == 1:
        customer_part = str(customers[0]).replace(' ', '_').replace('/', '_')[:20]
    elif len(customers) <= 3:
        customer_part = '_'.join([str(c).replace(' ', '_')[:10] for c in customers])[:30]
    else:
        customer_part = f"Multiple_{len(customers)}customers"
    
    invoices = data['Invoice No'].dropna().unique()
    if len(invoices) == 1:
        invoice_part = str(invoices[0]).replace(' ', '_')
    elif len(invoices) <= 3:
        invoice_part = '_'.join([str(i).replace(' ', '_') for i in invoices])[:20]
    else:
        invoice_part = f"Multiple_{len(invoices)}inv"
    
    orders = data['Sales Order Number'].dropna().unique()
    if len(orders) == 1:
        order_part = str(orders[0]).replace(' ', '_')
    elif len(orders) <= 3:
        order_part = '_'.join([str(o).replace(' ', '_') for o in orders])[:20]
    else:
        order_part = f"Multiple_{len(orders)}orders"
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    filename_parts = [
        size_spec,
        dpi_spec,
        customer_part,
        invoice_part,
        order_part,
        label_type.lower().replace(' ', '_'),
        timestamp
    ]
    
    filename = '_'.join(filename_parts) + '.zpl'
    
    if len(filename) > 200:
        filename = f"{size_spec}_{dpi_spec}_{customer_part[:15]}_{label_type.lower()}_{timestamp}.zpl"
    
    return filename

# =============================================================================
# PRINTER FUNCTIONS
# =============================================================================

def send_zpl_to_printer(zpl_data: str, printer_ip: str, 
                       printer_port: int = 9100) -> Tuple[bool, str]:
    """Send ZPL data to Zebra printer via network"""
    try:
        if not printer_ip:
            return False, "No printer IP address provided"
            
        socket.inet_aton(printer_ip)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.connect((printer_ip, printer_port))
            sock.send(zpl_data.encode('utf-8'))
            
        return True, "Label sent successfully"
        
    except socket.timeout:
        return False, "Connection timeout - check printer IP and network"
    except ConnectionRefusedError:
        return False, "Connection refused - check printer is on and IP is correct"
    except socket.gaierror:
        return False, "Invalid IP address format"
    except Exception as e:
        return False, f"Network error: {str(e)}"

# =============================================================================
# STREAMLIT UI
# =============================================================================

# Inject Browser Print detector at the start
inject_browser_print_detector()

st.sidebar.header("📊 Data Sources")

st.sidebar.subheader("📋 Sales Order")
sales_order_file = st.sidebar.file_uploader(
    "Choose Sales Order CSV", type=['csv'], key="sales_order_upload"
)

st.sidebar.subheader("📋 Package List")
packages_file = st.sidebar.file_uploader(
    "Choose Package List CSV", type=['csv'], key="packages_upload"
)

st.sidebar.subheader("📦 Products")
products_file = st.sidebar.file_uploader(
    "Choose Products CSV", type=['csv'], key="products_upload"
)

# Process button
if st.sidebar.button("🚀 Process Data", type="primary", 
                    disabled=not (sales_order_file and products_file and packages_file)):
    with st.spinner("Processing your data..."):
        sales_order_df = load_sales_order_csv(sales_order_file)
        products_df = load_standard_csv(products_file, "Products")
        packages_df = load_standard_csv(packages_file, "Packages")
        
        if sales_order_df is None or products_df is None or packages_df is None:
            st.error("Failed to load one or more CSV files")
            st.stop()
        
        file_info = (f"Sales Orders: {len(sales_order_df):,} rows | "
                    f"Products: {len(products_df):,} rows | "
                    f"Packages: {len(packages_df):,} rows")
        st.success("Files loaded successfully")
        st.info(file_info)
        
        processed_data = merge_data_sources(sales_order_df, products_df, packages_df)
        
        if processed_data is not None:
            st.session_state.processed_data = processed_data
            st.success(f"Successfully processed {len(processed_data):,} records")

# Version info at bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.caption(f"Version {VERSION}")
st.sidebar.caption("© 2025 DC Retail")

# Main Content
if st.session_state.processed_data is not None:
    processed_df = st.session_state.processed_data
    
    tab1, tab2 = st.tabs(["🎯 Generate Labels", "📊 Data Overview"])
    
    with tab1:
        st.header("🎯 Generate Labels")
        
        # Filters - Order: Customers, Delivery Dates, Sales Orders, Invoices
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            customers = sorted(processed_df['Customer'].dropna().unique().tolist())
            selected_customers = st.multiselect("Select Customers", customers)
            
        with col2:
            if selected_customers:
                filtered_dates = processed_df[
                    processed_df['Customer'].isin(selected_customers)
                ]['Delivery Date'].dropna().unique()
            else:
                filtered_dates = processed_df['Delivery Date'].dropna().unique()
            delivery_dates = sorted(filtered_dates.tolist())
            selected_dates = st.multiselect("Select Delivery Dates", delivery_dates)
            
        with col3:
            if selected_customers:
                filtered_orders = processed_df[
                    processed_df['Customer'].isin(selected_customers)
                ]['Sales Order Number'].dropna().unique()
            elif selected_dates:
                filtered_orders = processed_df[
                    processed_df['Delivery Date'].isin(selected_dates)
                ]['Sales Order Number'].dropna().unique()
            else:
                filtered_orders = processed_df['Sales Order Number'].dropna().unique()
            orders = sorted(filtered_orders.tolist())
            selected_orders = st.multiselect("Select Sales Orders", orders)
            
        with col4:
            if selected_customers:
                filtered_invoices = processed_df[
                    processed_df['Customer'].isin(selected_customers)
                ]['Invoice No'].dropna().unique()
            elif selected_dates:
                filtered_invoices = processed_df[
                    processed_df['Delivery Date'].isin(selected_dates)
                ]['Invoice No'].dropna().unique()
            elif selected_orders:
                filtered_invoices = processed_df[
                    processed_df['Sales Order Number'].isin(selected_orders)
                ]['Invoice No'].dropna().unique()
            else:
                filtered_invoices = processed_df['Invoice No'].dropna().unique()
            invoices = sorted(filtered_invoices.tolist())
            selected_invoices = st.multiselect("Select Invoices", invoices)
        
        # Apply filters
        filtered_df = processed_df.copy()
        
        if selected_customers:
            filtered_df = filtered_df[filtered_df['Customer'].isin(selected_customers)]
        if selected_dates:
            filtered_df = filtered_df[filtered_df['Delivery Date'].isin(selected_dates)]
        if selected_orders:
            filtered_df = filtered_df[filtered_df['Sales Order Number'].isin(selected_orders)]
        if selected_invoices:
            filtered_df = filtered_df[filtered_df['Invoice No'].isin(selected_invoices)]
        
        # Data display
        st.subheader(f"🏷️ Filtered Data ({len(filtered_df):,} records)")
        
        select_all = st.checkbox("Select all rows", value=True)
        
        if not select_all:
            selected_indices = st.multiselect(
                "Select specific rows to include:",
                options=filtered_df.index.tolist(),
                default=filtered_df.index.tolist(),
                format_func=lambda x: f"Row {x}: {filtered_df.loc[x, 'Product Name'] if 'Product Name' in filtered_df.columns else 'N/A'}"
            )
            display_data = filtered_df.loc[selected_indices] if selected_indices else pd.DataFrame()
        else:
            display_data = filtered_df
        
        if not display_data.empty:
            st.dataframe(display_data, use_container_width=True, height=400)
        else:
            st.warning("No data to display with current selection")
        
        # Label operations
        if not display_data.empty:
            st.header("🖨️ Label Operations")
            
            col1, col2, col3 = st.columns([3, 3, 4])
            
            with col1:
                label_type = st.selectbox(
                    "Label Type", 
                    ["Case", "Bin"],
                    help="Choose which type of labels to generate"
                )
            
            with col2:
                print_via = st.selectbox(
                    "Print Via",
                    ["Browser Print (Cloud)", "Download ZPL", "Network Printer (Local)"],
                    help="How do you want to output the labels?"
                )
            
            with col3:
                # Label size dropdown with TWO options
                label_size_options = ["1.75\" × 0.875\" (300 DPI)", "4\" × 2\" (203 DPI)"]
                selected_size = st.selectbox("Label Size", label_size_options)
                
                # Parse selected size
                if "4\" × 2\"" in selected_size:
                    label_width = 4.0
                    label_height = 2.0
                    dpi = 203
                else:
                    label_width = 1.75
                    label_height = 0.875
                    dpi = 300
                
                st.caption(f"DPI: {dpi}")
            
            # Calculate labels
            if label_type == "Case":
                total_labels = safe_sum(display_data['Case Labels Needed'])
            else:
                total_labels = safe_sum(display_data['Bin Labels Needed'])
            
            if total_labels > 0:
                st.markdown("---")
                
                col1, col2, col3 = st.columns([2, 2, 6])
                with col1:
                    st.metric("Total Labels", f"{int(total_labels)}")
                with col2:
                    st.metric("Label Type", label_type)
                
                st.markdown("### Actions")
                
                # Different UI based on print method
                if print_via == "Browser Print (Cloud)":
                    # Browser Print Interface
                    st.info("🔌 **Zebra Browser Print** enables direct printing from your browser to local printers")
                    
                    with st.expander("📋 Setup Instructions", expanded=False):
                        st.markdown("""
                        **One-time setup:**
                        1. Download and install [Zebra Browser Print](https://www.zebra.com/us/en/support-downloads/software/printer-software/browser-print.html)
                        2. Ensure your Zebra printer is connected (USB or network)
                        3. Browser Print will automatically detect your printer
                        4. Click the Print button below to send labels directly to your printer
                        
                        **Troubleshooting:**
                        - Make sure Browser Print is running (check system tray)
                        - Refresh this page after installing Browser Print
                        - For network printers, ensure printer is on the same network
                        """)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button(f"🖨️ Print {int(total_labels)} Labels via Browser Print", 
                                   type="primary", use_container_width=True):
                            try:
                                with st.spinner(f"Sending {int(total_labels)} labels to printer..."):
                                    labels = generate_labels_for_dataset(display_data, label_type, 
                                                                       label_width, label_height, dpi)
                                    
                                    if not labels:
                                        st.error("No labels were generated")
                                    else:
                                        # Combine all labels into one ZPL string
                                        combined_zpl = "\n".join(labels)
                                        
                                        # Send to Browser Print
                                        send_to_browser_print(combined_zpl)
                                        
                                        st.success(f"Sent {len(labels)} labels to Browser Print")
                                        st.info("Check your printer - labels should be printing!")
                            except Exception as e:
                                st.error(f"Error during printing: {str(e)}")
                    
                    with col2:
                        if st.button("👀 Preview Sample ZPL", use_container_width=True):
                            try:
                                if len(display_data) > 0:
                                    sample_row = display_data.iloc[0]
                                    sample_labels = generate_all_labels_for_row(sample_row, label_type, 
                                                                              label_width, label_height, dpi)
                                    
                                    if sample_labels:
                                        with st.expander("ZPL Preview", expanded=True):
                                            st.code(sample_labels[0], language="text")
                                        if len(sample_labels) > 1:
                                            st.info(f"This product generates {len(sample_labels)} labels")
                                    else:
                                        st.warning("No labels generated for the first product")
                            except Exception as e:
                                st.error(f"Error generating preview: {str(e)}")
                
                elif print_via == "Network Printer (Local)":
                    # Network Printer Interface (existing)
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        printer_ip = st.text_input(
                            "Printer IP", 
                            placeholder="192.168.1.100",
                            help="Enter the IP address of your Zebra printer"
                        )
                        
                        if printer_ip:
                            if st.button(f"🖨️ Print {int(total_labels)} Labels", type="primary", use_container_width=True):
                                try:
                                    with st.spinner(f"Printing {int(total_labels)} labels..."):
                                        labels = generate_labels_for_dataset(display_data, label_type, 
                                                                           label_width, label_height, dpi)
                                        
                                        if not labels:
                                            st.error("No labels were generated")
                                        else:
                                            success_count = 0
                                            for i, zpl in enumerate(labels, 1):
                                                success, message = send_zpl_to_printer(zpl, printer_ip)
                                                if success:
                                                    success_count += 1
                                                    if i % 20 == 0:
                                                        st.write(f"Printed {i}/{len(labels)} labels...")
                                                else:
                                                    st.error(f"Printing failed at label {i}: {message}")
                                                    break
                                            
                                            if success_count == len(labels):
                                                st.success(f"Successfully printed {success_count} labels")
                                            elif success_count > 0:
                                                st.warning(f"Printed {success_count} of {len(labels)} labels")
                                except Exception as e:
                                    st.error(f"Error during printing: {str(e)}")
                        else:
                            st.button(f"🖨️ Print {int(total_labels)} Labels", disabled=True, use_container_width=True)
                            st.info("Enter printer IP address above")
                    
                    with col2:
                        st.write("")
                        if st.button("👀 Preview Sample ZPL", use_container_width=True):
                            try:
                                if len(display_data) > 0:
                                    sample_row = display_data.iloc[0]
                                    sample_labels = generate_all_labels_for_row(sample_row, label_type, 
                                                                              label_width, label_height, dpi)
                                    
                                    if sample_labels:
                                        st.code(sample_labels[0], language="text")
                                        if len(sample_labels) > 1:
                                            st.info(f"This product generates {len(sample_labels)} labels")
                                    else:
                                        st.warning("No labels generated for the first product")
                            except Exception as e:
                                st.error(f"Error generating preview: {str(e)}")
                
                else:  # Download ZPL
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button(f"📥 Generate {int(total_labels)} Labels", type="primary", use_container_width=True):
                            try:
                                with st.spinner("Generating ZPL..."):
                                    labels = generate_labels_for_dataset(display_data, label_type, 
                                                                       label_width, label_height, dpi)
                                    
                                    if not labels:
                                        st.error("No labels were generated")
                                    else:
                                        zpl_content = "\n".join(labels)
                                        filename = generate_filename(display_data, label_type, 
                                                                   label_width, label_height, dpi)
                                        
                                        st.session_state[f'zpl_content_{label_type}'] = zpl_content
                                        st.session_state[f'zpl_filename_{label_type}'] = filename
                                        st.success(f"Generated {len(labels)} labels")
                            except Exception as e:
                                st.error(f"Error during generation: {str(e)}")
                        
                        if st.button("👀 Preview Sample ZPL", use_container_width=True):
                            try:
                                if len(display_data) > 0:
                                    sample_row = display_data.iloc[0]
                                    sample_labels = generate_all_labels_for_row(sample_row, label_type, 
                                                                              label_width, label_height, dpi)
                                    
                                    if sample_labels:
                                        st.code(sample_labels[0], language="text")
                                        if len(sample_labels) > 1:
                                            st.info(f"This product generates {len(sample_labels)} labels")
                                    else:
                                        st.warning("No labels generated for the first product")
                            except Exception as e:
                                st.error(f"Error generating preview: {str(e)}")
                    
                    with col2:
                        zpl_key = f'zpl_content_{label_type}'
                        filename_key = f'zpl_filename_{label_type}'
                        
                        if zpl_key in st.session_state and st.session_state[zpl_key]:
                            st.download_button(
                                label=f"💾 Download {label_type} ZPL",
                                data=st.session_state[zpl_key],
                                file_name=st.session_state[filename_key],
                                mime="text/plain",
                                use_container_width=True
                            )
                        else:
                            st.button(f"💾 Download {label_type} ZPL", disabled=True, use_container_width=True)
                            st.info("Generate labels first")
            
            else:
                st.warning(f"No {label_type.lower()} labels needed for selected data")
            
            # CSV Export
            st.header("💾 Export Data")
            
            csv_filename = generate_filename(display_data, "data", label_width, label_height, dpi).replace('.zpl', '.csv')
            
            csv_buffer = io.StringIO()
            display_data.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📄 Download Filtered Data CSV",
                data=csv_buffer.getvalue(),
                file_name=csv_filename,
                mime="text/csv"
            )
    
    with tab2:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Sales Orders", len(processed_df['Sales Order Number'].dropna().unique()))
        with col2:
            st.metric("Customers", len(processed_df['Customer'].dropna().unique()))
        with col3:
            st.metric("Total Items", len(processed_df))
        with col4:
            st.metric("Categories", len(processed_df['Category'].dropna().unique()))
        with col5:
            case_items = safe_count_nonzero(processed_df['Case Labels Needed'])
            bin_items = safe_count_nonzero(processed_df['Bin Labels Needed'])
            coverage = max(case_items, bin_items) / len(processed_df) * 100 if len(processed_df) > 0 else 0
            st.metric("Label Coverage", f"{coverage:.0f}%")
        
        st.subheader("📈 Category Breakdown")
        category_counts = processed_df['Category'].value_counts()
        st.bar_chart(category_counts)
        
        st.subheader("👥 Customer Breakdown")
        customer_counts = processed_df['Customer'].value_counts().head(10)
        st.bar_chart(customer_counts)
        
        st.subheader("🔍 Complete Dataset")
        st.dataframe(processed_df, use_container_width=True)

else:
    if not sales_order_file and not products_file and not packages_file:
        st.info("Upload the required CSV files in the sidebar to get started")
        
        with st.expander("ℹ️ How it Works", expanded=True):
            st.markdown("""
            **Key Features:**
            - Links Sales Orders with Products and Packages
            - Calculates Case/Bin Labels automatically with partial quantity support
            - Multiple label sizes: 1.75" × 0.875" (300 DPI) and 4" × 2" (203 DPI)
            - **NEW: Browser Print support for cloud printing!**
            - Direct Zebra printer support or ZPL download
            - Smart file naming with label specs, customer, invoice, and order info
            
            **Printing Options:**
            - **Browser Print (Cloud):** Print directly from your browser to local Zebra printers
            - **Network Printer (Local):** Direct IP connection when running locally
            - **Download ZPL:** Save files for manual printing or batch processing
            
            **Label Size Support:**
            - Small labels: 1.75" × 0.875" at 300 DPI (ZD410)
            - Large labels: 4" × 2" at 203 DPI (ZD621)
            - Automatic proportional scaling maintains layout consistency
            """)
    
    elif sales_order_file and products_file and packages_file:
        st.info("Click the 'Process Data' button in the sidebar to analyze your files")
    
    else:
        missing_files = []
        if not sales_order_file:
            missing_files.append("Sales Order")
        if not products_file:
            missing_files.append("Products")
        if not packages_file:
            missing_files.append("Package List")
        
        st.warning(f"Please upload the {' and '.join(missing_files)} CSV file(s) to continue")

if not QR_AVAILABLE:
    st.error("QR code libraries not available. Install with: pip install qrcode[pil]")