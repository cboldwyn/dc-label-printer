"""
DC Retail Case & Bin Label Generator - Cloud-Safe Version
========================================================

Version 2.6.3 - Optimized for Streamlit Cloud deployment
- Fixes CSP issues with Browser Print integration
- Default to Bin labels, Download ZPL, and 4" × 2" size
- Labels grouped by Invoice, then sorted A-Z by Product Name
- Added 2.625" × 1" label size option for ZD621
- Brand/Product split using first hyphen
- Brand displayed with inverted colors (white on black)
- Category right-aligned on brand bar
- Narrower box for quantities with smaller Pkg Qty font
- Fixed text positioning within quantity box

Author: DC Retail
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
VERSION = "2.6.3"

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
    session_vars = ['processed_data', 'sales_order_data', 'products_data', 'packages_data']
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

initialize_session_state()

# =============================================================================
# BROWSER PRINT INTEGRATION (CLOUD-SAFE VERSION)
# =============================================================================

def create_browser_print_launcher(zpl_data: str, label_count: int) -> None:
    """Create a launcher for Browser Print that works with Streamlit Cloud CSP"""
    
    # Encode ZPL data as base64 for safe transport
    b64_zpl = base64.b64encode(zpl_data.encode()).decode()
    
    html_content = f"""
    <div style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;">
        <h3 style="color: #333; margin-top: 0;">🖨️ Ready to Print {label_count} Labels</h3>
        
        <p style="color: #666;">Choose your printing method:</p>
        
        <div style="display: flex; gap: 10px; margin-top: 15px;">
            <!-- Method 1: Direct Browser Print (if installed) -->
            <button onclick="sendToBrowserPrint()" style="
                background-color: #4CAF50;
                color: white;
                padding: 12px 20px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                flex: 1;
            ">
                🚀 Send to Browser Print
            </button>
            
            <!-- Method 2: Copy to Clipboard -->
            <button onclick="copyToClipboard()" style="
                background-color: #2196F3;
                color: white;
                padding: 12px 20px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                flex: 1;
            ">
                📋 Copy ZPL to Clipboard
            </button>
        </div>
        
        <div id="status" style="margin-top: 15px; padding: 10px; display: none;"></div>
        
        <details style="margin-top: 20px;">
            <summary style="cursor: pointer; color: #666;">ℹ️ Help & Instructions</summary>
            <div style="margin-top: 10px; padding: 10px; background-color: #fff; border-radius: 5px;">
                <strong>Option 1: Browser Print (Recommended)</strong>
                <ol style="margin: 10px 0;">
                    <li>Install <a href="https://www.zebra.com/us/en/support-downloads/software/printer-software/browser-print.html" target="_blank">Zebra Browser Print</a></li>
                    <li>Make sure it's running (check system tray)</li>
                    <li>Click "Send to Browser Print"</li>
                </ol>
                
                <strong>Option 2: Copy & Paste</strong>
                <ol style="margin: 10px 0;">
                    <li>Click "Copy ZPL to Clipboard"</li>
                    <li>Open Zebra Setup Utilities or Notepad</li>
                    <li>Paste and send to printer</li>
                </ol>
            </div>
        </details>
    </div>
    
    <script>
        // Store the ZPL data
        const zplData = atob('{b64_zpl}');
        
        function sendToBrowserPrint() {{
            const statusDiv = document.getElementById('status');
            statusDiv.style.display = 'block';
            statusDiv.style.backgroundColor = '#fff3cd';
            statusDiv.style.color = '#856404';
            statusDiv.innerHTML = '⏳ Attempting to connect to Browser Print...';
            
            // Try to open Browser Print endpoint
            const printWindow = window.open('http://localhost:9100', 'browserPrint', 'width=1,height=1,left=-100,top=-100');
            
            if (printWindow) {{
                setTimeout(() => {{
                    try {{
                        // Try to send data via postMessage
                        printWindow.postMessage({{type: 'print', zpl: zplData}}, 'http://localhost:9100');
                        
                        // Also try form submission as fallback
                        const form = document.createElement('form');
                        form.method = 'POST';
                        form.action = 'http://localhost:9100/print';
                        form.target = 'browserPrint';
                        
                        const input = document.createElement('textarea');
                        input.name = 'zpl';
                        input.value = zplData;
                        form.appendChild(input);
                        
                        document.body.appendChild(form);
                        form.submit();
                        document.body.removeChild(form);
                        
                        statusDiv.style.backgroundColor = '#d4edda';
                        statusDiv.style.color = '#155724';
                        statusDiv.innerHTML = '✅ Sent to Browser Print! Check your printer.';
                        
                        setTimeout(() => {{
                            printWindow.close();
                        }}, 2000);
                    }} catch (e) {{
                        statusDiv.style.backgroundColor = '#f8d7da';
                        statusDiv.style.color = '#721c24';
                        statusDiv.innerHTML = '❌ Could not connect. Is Browser Print installed and running?';
                        printWindow.close();
                    }}
                }}, 1000);
            }} else {{
                statusDiv.style.backgroundColor = '#f8d7da';
                statusDiv.style.color = '#721c24';
                statusDiv.innerHTML = '❌ Could not open Browser Print. Please check if it is installed.';
            }}
        }}
        
        function copyToClipboard() {{
            const statusDiv = document.getElementById('status');
            
            navigator.clipboard.writeText(zplData).then(() => {{
                statusDiv.style.display = 'block';
                statusDiv.style.backgroundColor = '#d4edda';
                statusDiv.style.color = '#155724';
                statusDiv.innerHTML = '✅ ZPL copied to clipboard! Paste it into Zebra Setup Utilities.';
            }}).catch(() => {{
                // Fallback for older browsers
                const textarea = document.createElement('textarea');
                textarea.value = zplData;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                
                statusDiv.style.display = 'block';
                statusDiv.style.backgroundColor = '#d4edda';
                statusDiv.style.color = '#155724';
                statusDiv.innerHTML = '✅ ZPL copied to clipboard! Paste it into Zebra Setup Utilities.';
            }});
        }}
    </script>
    """
    
    components.html(html_content, height=300)

# =============================================================================
# UTILITY FUNCTIONS (keeping all the same from original)
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
    """Generate ZPL code for Zebra printer labels with brand extraction and inverted display."""
    width_dots = int(label_width * dpi)
    height_dots = int(label_height * dpi)
    
    # Calculate scaling factor based on WIDTH primarily (not average)
    reference_width = 4.0 * 203  # Using 4" × 2" at 203 DPI as reference
    reference_height = 2.0 * 203
    scale_x = width_dots / reference_width
    scale_y = height_dots / reference_height
    
    # Font sizes - scaled primarily by width to maintain readability
    fonts = {
        'extra_large': int(38 * scale_x),
        'large': int(32 * scale_x),
        'medium': int(24 * scale_x),
        'large_plus': int(28 * scale_x),
        'small': int(20 * scale_x),
        'small_plus': int(22 * scale_x)
    }
    
    # Layout constants
    left_margin = int(20 * scale_x)  # Reduced margin to use more space
    right_margin = int(20 * scale_x)
    
    # Line spacing
    line_spacing = int(38 * scale_y)  # Space between lines
    
    # Vertical positions - ADJUSTED for new layout
    # Brand bar at top, Product below, Batch below that
    brand_bar_y = int(height_dots * 0.02)  # Start black bar very close to top
    brand_bar_height = int(height_dots * 0.12)  # Height of black bar
    product_y = int(height_dots * 0.17)  # Product below black bar with gap
    batch_y = int(height_dots * 0.27)  # Batch below product
    
    # Center positions for quantity box
    qty_box_y = int(height_dots * 0.38)  # Start of qty box
    qty_box_height = int(height_dots * 0.16)  # Height of qty box (reduced from 0.18)
    
    # Positions within the box - adjusted for better spacing
    bin_qty_y = qty_box_y + int(qty_box_height * 0.25)  # Bin Qty in upper half (adjusted)
    separator_y = qty_box_y + int(qty_box_height * 0.5)  # Middle line
    pkg_qty_y = qty_box_y + int(qty_box_height * 0.68)  # Pkg Qty in lower half (adjusted)
    
    delivered_y = int(height_dots * 0.65)
    sell_by_y = int(height_dots * 0.75)
    
    qr_data = sanitize_qr_data(package_label)
    
    # QR code positioning - in the middle right area
    qr_size = max(4, int(5 * min(scale_x, scale_y)))
    qr_box_size = qr_size * 30  # Approximate QR code box size in dots
    
    # Position QR in right side, middle height area
    qr_right_margin = int(15 * scale_x)
    qr_x = width_dots - qr_box_size - qr_right_margin
    qr_y = int(height_dots * 0.30)  # Adjusted QR position
    
    # Calculate available width for text
    text_available_width = width_dots - left_margin - right_margin
    
    # Calculate characters that fit based on font and available width
    max_chars_brand = int(text_available_width / (fonts['large'] * 0.45))
    max_chars_product = int(text_available_width / (fonts['large'] * 0.45))
    
    # EXTRACT BRAND AND PRODUCT from product_name using FIRST hyphen only
    product_name_str = str(product_name) if pd.notna(product_name) else ""
    brand = ""
    product = product_name_str  # Default to full name if no hyphen
    
    if ' - ' in product_name_str:
        # Split ONLY on the FIRST hyphen
        parts = product_name_str.split(' - ', 1)  # maxsplit=1 ensures only first hyphen is used
        brand = parts[0].strip()
        product = parts[1].strip()  # This may contain additional hyphens, which is fine
    elif '-' in product_name_str:
        # Handle case without spaces around hyphen
        parts = product_name_str.split('-', 1)  # maxsplit=1 ensures only first hyphen is used
        brand = parts[0].strip()
        product = parts[1].strip()
    
    # Truncate brand if too long
    if len(brand) > max_chars_brand:
        brand = brand[:max_chars_brand-3] + "..."
    
    # Truncate product if too long (single line only now)
    if len(product) > max_chars_product:
        product = product[:max_chars_product-3] + "..."
    
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
    
    # Combined bottom line - ALSO uses full width
    max_chars_bottom = int(text_available_width / (fonts['small_plus'] * 0.45))
    
    combined_parts = []
    if invoice_no:
        combined_parts.append(str(invoice_no))
    if metrc_manifest:
        combined_parts.append(str(metrc_manifest))
    if qr_data:
        # Don't truncate the UID unless absolutely necessary
        combined_parts.append(qr_data)
    
    combined_text = " | ".join(combined_parts)
    
    # Only truncate if exceeding the calculated max for FULL width
    if len(combined_text) > max_chars_bottom:
        combined_text = combined_text[:max_chars_bottom-3] + "..."
    
    # Build ZPL
    zpl_lines = ["^XA"]
    
    # INVERTED BRAND BAR - white text on black background
    # Draw black rectangle for brand background (full width)
    zpl_lines.append(f"^FO0,{brand_bar_y}^GB{width_dots},{brand_bar_height},{brand_bar_height}^FS")
    
    # Calculate vertical center of text within black bar
    brand_text_y = brand_bar_y + int((brand_bar_height - fonts['large']) / 2)
    
    # White text for brand (using reverse field)
    zpl_lines.append("^FR")  # Start field reverse
    zpl_lines.append(f"^CF0,{fonts['large']}")
    zpl_lines.append(f"^FO{left_margin},{brand_text_y}^FR^FD{brand}^FS")
    
    # CATEGORY - right aligned on same row as brand (also white on black)
    category_text = str(category) if pd.notna(category) else ""
    if category_text:
        # Calculate position for right-aligned text
        # Estimate text width (rough approximation)
        category_width = len(category_text) * int(fonts['medium'] * 0.5)
        category_x = width_dots - right_margin - category_width
        category_text_y = brand_bar_y + int((brand_bar_height - fonts['medium']) / 2)
        
        zpl_lines.append(f"^CF0,{fonts['medium']}")
        zpl_lines.append(f"^FO{category_x},{category_text_y}^FR^FD{category_text}^FS")
    
    # Product name - below brand bar in normal black text
    zpl_lines.append(f"^CF0,{fonts['large']}")
    zpl_lines.append(f"^FO{left_margin},{product_y}^FD{product}^FS")
    
    # Batch - positioned below product name
    zpl_lines.extend([
        f"^CF0,{fonts['medium']}",
        f"^FO{left_margin},{batch_y}^FDBatch: {batch_no}^FS"
    ])
    
    # QUANTITY BOX - Draw box around quantities
    # Calculate box dimensions - NARROWER box
    box_width = int(width_dots * 0.28)  # Box takes only 28% of label width (was 45%)
    box_x = int((width_dots - box_width) / 2)  # Center the box horizontally
    
    # Draw outer box (border thickness 2)
    zpl_lines.append(f"^FO{box_x},{qty_box_y}^GB{box_width},{qty_box_height},2^FS")
    
    # Draw horizontal separator line in middle of box
    zpl_lines.append(f"^FO{box_x},{separator_y}^GB{box_width},2,2^FS")
    
    # BIN/CASE QUANTITY - in upper half of box
    if qty == int(qty):
        qty_display = f"{label_type} Qty: {int(qty)}"
    else:
        qty_display = f"{label_type} Qty: {qty:.1f}"
    
    # Center the text horizontally within the box
    qty_text_width = len(qty_display) * int(fonts['large_plus'] * 0.5)
    qty_text_x = box_x + int((box_width - qty_text_width) / 2)
    
    zpl_lines.extend([
        f"^CF0,{fonts['large_plus']}",
        f"^FO{qty_text_x},{bin_qty_y}^FD{qty_display}^FS"
    ])
    
    # PACKAGE QUANTITY - in lower half of box with SMALLER font
    if pkg_qty:
        pkg_display = f"Pkg Qty: {pkg_qty}"
        # Use smaller font for package quantity
        pkg_font_size = fonts['medium']  # Changed from 'large_plus' to 'medium'
        pkg_text_width = len(pkg_display) * int(pkg_font_size * 0.5)
        pkg_text_x = box_x + int((box_width - pkg_text_width) / 2)
        
        zpl_lines.extend([
            f"^CF0,{pkg_font_size}",
            f"^FO{pkg_text_x},{pkg_qty_y}^FD{pkg_display}^FS"
        ])
    
    # Delivered - fixed position
    zpl_lines.extend([
        f"^CF0,{fonts['large_plus']}",
        f"^FO{left_margin},{delivered_y}^FDDelivered: {formatted_date}^FS"
    ])
    
    # Sell by on its own line
    if sell_by_formatted:
        zpl_lines.extend([
            f"^CF0,{fonts['large_plus']}",
            f"^FO{left_margin},{sell_by_y}^FDSell By: {sell_by_formatted}^FS"
        ])
    
    # QR code - positioned in middle right (doesn't interfere with top or bottom text)
    if qr_data:
        zpl_lines.append(f"^FO{qr_x},{qr_y}^BQN,2,{qr_size}^FDQA,{qr_data}^FS")
    
    # Combined bottom line - at the very bottom using FULL WIDTH
    bottom_line_y = int(height_dots * 0.88)
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
    """Generate all labels for entire dataset, grouped by Invoice then sorted A-Z by Product Name"""
    all_labels = []
    
    # First, sort by Invoice No (nulls last), then by Product Name A-Z
    df_sorted = df.sort_values(['Invoice No', 'Product Name'], 
                               ascending=[True, True], 
                               na_position='last')
    
    # Group by Invoice No to keep invoice items together
    for invoice_no, invoice_group in df_sorted.groupby('Invoice No', dropna=False):
        # Within each invoice group, items are already sorted A-Z by Product Name
        for _, row in invoice_group.iterrows():
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
# STREAMLIT UI - Main Application
# =============================================================================

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
        
        # Filters
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
                    ["Bin", "Case"],  # Bin is now first (default)
                    help="Choose which type of labels to generate"
                )
            
            with col2:
                print_via = st.selectbox(
                    "Print Via",
                    ["Download ZPL", "Browser Print (Cloud)", "Network Printer (Local)"],  # Download ZPL is now first (default)
                    help="How do you want to output the labels?"
                )
            
            with col3:
                label_size_options = [
                    "4\" × 2\" (203 DPI)",
                    "2.625\" × 1\" (203 DPI)",  # NEW SIZE OPTION
                    "1.75\" × 0.875\" (300 DPI)"
                ]
                selected_size = st.selectbox("Label Size", label_size_options)
                
                # Parse selected size
                if "4\" × 2\"" in selected_size:
                    label_width = 4.0
                    label_height = 2.0
                    dpi = 203
                elif "2.625\" × 1\"" in selected_size:
                    label_width = 2.625
                    label_height = 1.0
                    dpi = 203
                else:  # 1.75" × 0.875"
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
                if print_via == "Download ZPL":  # This is now the default/first option
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
                
                elif print_via == "Browser Print (Cloud)":
                    if st.button(f"🖨️ Generate {int(total_labels)} Labels for Browser Print", 
                               type="primary", use_container_width=True):
                        try:
                            with st.spinner(f"Generating {int(total_labels)} labels..."):
                                labels = generate_labels_for_dataset(display_data, label_type, 
                                                                   label_width, label_height, dpi)
                                
                                if not labels:
                                    st.error("No labels were generated")
                                else:
                                    combined_zpl = "\n".join(labels)
                                    st.success(f"✅ Generated {len(labels)} labels!")
                                    
                                    # Use the cloud-safe launcher
                                    create_browser_print_launcher(combined_zpl, len(labels))
                                    
                                    # Also provide direct download option
                                    st.markdown("### Alternative: Direct Download")
                                    filename = generate_filename(display_data, label_type, 
                                                               label_width, label_height, dpi)
                                    st.download_button(
                                        label=f"💾 Download ZPL File",
                                        data=combined_zpl,
                                        file_name=filename,
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                                    
                        except Exception as e:
                            st.error(f"Error during generation: {str(e)}")
                
                elif print_via == "Network Printer (Local)":
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
            - **NEW: Brand/Product Extraction** - Automatically splits product names at first hyphen
            - **NEW: Inverted Brand Display** - Brand shown with white text on black background
            - Links Sales Orders with Products and Packages
            - Calculates Case/Bin Labels automatically with partial quantity support
            - Multiple label sizes: 
                - 4" × 2" at 203 DPI (ZD621 - large labels)
                - 2.625" × 1" at 203 DPI (ZD621 - medium labels)
                - 1.75" × 0.875" at 300 DPI (ZD410 - small labels)
            - Browser Print support for cloud printing
            - Direct Zebra printer support or ZPL download
            - Smart file naming with label specs, customer, invoice, and order info
            
            **Brand/Product Format:**
            - Products like "Camino - Strawberry Sunset Sours Gummies 100mg"
            - Become: Brand = "Camino", Product = "Strawberry Sunset Sours Gummies 100mg"
            - Only the FIRST hyphen is used as separator
            - Brand displays with inverted colors (white on black)
            
            **Printing Options:**
            - **Download ZPL:** Save files for manual printing or batch processing (default)
            - **Browser Print (Cloud):** Print directly from your browser to local Zebra printers
            - **Network Printer (Local):** Direct IP connection when running locally
            
            **Sorting:**
            - Labels are grouped by Invoice Number
            - Within each invoice, products are sorted A-Z alphabetically
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