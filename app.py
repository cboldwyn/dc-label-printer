"""
Haven Cannabis Label Data Processor v1.0
=========================================

A Streamlit application for processing CSV files and generating Zebra printer labels.
Supports direct printing to Zebra ZD410/ZD621 printers via network or USB connection.

Author: Haven Cannabis
Created: 2025
"""

import streamlit as st
import pandas as pd
import io
import math
import socket
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# Import QR code libraries with error handling
try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    st.error("QR code libraries not installed. Run: pip install qrcode[pil]")

# Page configuration
st.set_page_config(
    page_title="Haven Cannabis - Label Data Processor",
    page_icon="🏷️",
    layout="wide"
)

# App header
st.title("🏷️ Label Data Processor v1.0")
st.markdown("**Haven Cannabis** | Process CSV files for label printing with calculated quantities")

# Session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = [
        'processed_data',
        'sales_order_data', 
        'products_data',
        'packages_data'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

initialize_session_state()

# =============================================================================
# FILE UPLOAD SECTION
# =============================================================================

st.sidebar.header("📊 Data Sources")

# Sales Order CSV Upload
st.sidebar.subheader("📋 Sales Order")
sales_order_file = st.sidebar.file_uploader(
    "Choose Sales Order CSV",
    type=['csv'],
    key="sales_order_upload",
    label_visibility="collapsed"
)

# Package List CSV Upload
st.sidebar.subheader("📋 Package List")
packages_file = st.sidebar.file_uploader(
    "Choose Package List CSV",
    type=['csv'],
    key="packages_upload", 
    label_visibility="collapsed"
)

# Products CSV Upload  
st.sidebar.subheader("📦 Products")
products_file = st.sidebar.file_uploader(
    "Choose Products CSV",
    type=['csv'],
    key="products_upload",
    label_visibility="collapsed"
)

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def load_sales_order_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load Sales Order CSV with special handling for metadata lines.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        # Try different parsing methods for robustness
        parsing_methods = [
            {'skiprows': 3},
            {'skiprows': 3, 'encoding': 'utf-8', 'quotechar': '"', 'skipinitialspace': True},
            {'skiprows': 3, 'sep': ';', 'encoding': 'utf-8'},
            {'skiprows': 3, 'sep': '\t', 'encoding': 'utf-8'}
        ]
        
        for method in parsing_methods:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                return pd.read_csv(uploaded_file, **method)
            except Exception:
                continue
                
        return None
        
    except Exception as e:
        st.error(f"Error loading Sales Order CSV: {str(e)}")
        return None

def load_standard_csv(uploaded_file, file_type: str) -> Optional[pd.DataFrame]:
    """
    Load standard CSV files (Products, Packages).
    
    Args:
        uploaded_file: Streamlit uploaded file object
        file_type: Type of file for error messaging
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading {file_type} CSV: {str(e)}")
        return None

def format_delivery_date(date_series: pd.Series) -> pd.Series:
    """
    Format delivery dates to mm-dd-yy format.
    
    Args:
        date_series: Pandas series containing dates
        
    Returns:
        Formatted date series
    """
    try:
        # Convert to datetime first
        date_series = pd.to_datetime(date_series, errors='coerce')
        # Format as mm-dd-yy
        return date_series.dt.strftime('%m-%d-%y')
    except Exception:
        return date_series.astype(str)

def calculate_label_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Case Labels Needed and Bin Labels Needed columns.
    
    Args:
        df: DataFrame with Package Quantity, Case Quantity, Bin Quantity columns
        
    Returns:
        DataFrame with added calculated columns
    """
    # Case Labels Needed = ceil(Package Quantity / Case Quantity)
    if 'Package Quantity' in df.columns and 'Case Quantity' in df.columns:
        df['Case Labels Needed'] = df.apply(
            lambda row: math.ceil(row['Package Quantity'] / row['Case Quantity']) 
            if (pd.notna(row['Package Quantity']) and 
                pd.notna(row['Case Quantity']) and 
                row['Case Quantity'] > 0) 
            else 0, 
            axis=1
        )
    else:
        df['Case Labels Needed'] = 0
    
    # Bin Labels Needed = ceil(Package Quantity / Bin Quantity) 
    if 'Package Quantity' in df.columns and 'Bin Quantity' in df.columns:
        df['Bin Labels Needed'] = df.apply(
            lambda row: math.ceil(row['Package Quantity'] / row['Bin Quantity']) 
            if (pd.notna(row['Package Quantity']) and 
                pd.notna(row['Bin Quantity']) and 
                row['Bin Quantity'] > 0) 
            else 0, 
            axis=1
        )
    else:
        df['Bin Labels Needed'] = 0
    
    return df

def merge_data_sources(sales_order_df: pd.DataFrame, 
                      products_df: pd.DataFrame, 
                      packages_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merge the three data sources and create the final label dataset.
    
    Args:
        sales_order_df: Sales order data
        products_df: Products data  
        packages_df: Packages data
        
    Returns:
        Merged and processed DataFrame or None if error
    """
    try:
        # Store raw data in session state
        st.session_state.sales_order_data = sales_order_df
        st.session_state.products_data = products_df
        st.session_state.packages_data = packages_df
        
        # First merge: Sales Order + Products
        merged_df = sales_order_df.merge(
            products_df, 
            left_on='Product Id', 
            right_on='ID', 
            how='left',
            suffixes=('', '_products')
        )
        
        # Second merge: + Packages
        final_df = merged_df.merge(
            packages_df, 
            left_on='Package Label', 
            right_on='Package Label', 
            how='left',
            suffixes=('', '_packages')
        )
        
        # Column mapping for cleaner output
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
        
        # Create final DataFrame with mapped columns
        label_data = pd.DataFrame()
        
        for original_col, new_col in column_mapping.items():
            if original_col in final_df.columns:
                label_data[new_col] = final_df[original_col]
            else:
                label_data[new_col] = None
        
        # Format delivery dates
        if 'Delivery Date' in label_data.columns:
            label_data['Delivery Date'] = format_delivery_date(label_data['Delivery Date'])
        
        # Calculate label quantities
        label_data = calculate_label_quantities(label_data)
        
        # Reorder columns for consistent output
        desired_order = [
            'Product Name', 'Category', 'Batch No', 'Package Label', 
            'Package Quantity', 'Case Quantity', 'Bin Quantity', 
            'Case Labels Needed', 'Bin Labels Needed',
            'Customer', 'Invoice No', 'Sales Order Number', 'METRC Manifest', 
            'Delivery Date', 'Sell by'
        ]
        
        return label_data[desired_order]
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        show_column_debug_info(sales_order_df, products_df, packages_df)
        return None

def show_column_debug_info(sales_order_df: pd.DataFrame, 
                          products_df: pd.DataFrame, 
                          packages_df: pd.DataFrame):
    """Show debug information about CSV column structure"""
    st.write("**Debug: Column Structure**")
    st.write("**Sales Order columns:**", list(sales_order_df.columns))
    st.write("**Products columns:**", list(products_df.columns))
    st.write("**Packages columns:**", list(packages_df.columns))

# =============================================================================
# ZEBRA PRINTING FUNCTIONS  
# =============================================================================

def generate_label_zpl(product_name: str, batch_no: str, qty: str, 
                      pkg_qty: str, date_str: str, package_label: str, 
                      sell_by: str, invoice_no: str, sales_order: str, category: str,
                      label_type: str = "Case",
                      label_width: float = 1.75, label_height: float = 0.875, 
                      dpi: int = 300) -> str:
    """
    Generate ZPL code for Zebra printer labels with flexible sizing.
    
    Args:
        product_name: Product name (will be truncated if too long)
        batch_no: Batch number
        qty: Quantity (Case Qty or Bin Qty depending on label type)
        pkg_qty: Package quantity
        date_str: Date string
        package_label: Package label for QR code
        sell_by: Sell by date
        invoice_no: Invoice number
        sales_order: Sales order number
        category: Product category
        label_type: "Case" or "Bin" for label formatting
        label_width: Label width in inches
        label_height: Label height in inches
        dpi: Printer DPI (dots per inch)
        
    Returns:
        ZPL code string
    """
    # Calculate label dimensions in dots
    width_dots = int(label_width * dpi)
    height_dots = int(label_height * dpi)
    
    # Font sizes for 300 DPI labels
    extra_large_font = 32  # Qty
    large_font = 28        # Product name, Delivered/Date, Category  
    medium_font = 20       # Batch, Pkg Qty, Invoice/SO
    small_font = 16        # Package label text at bottom
    
    # Calculate margins and positioning
    left_margin = 30
    top_margin = 20
    
    # Handle product name - can use full width
    product_name = str(product_name) if pd.notna(product_name) else ""
    if len(product_name) > 35:
        words = product_name.split()
        line1 = ""
        line2 = ""
        
        for word in words:
            if len(line1 + " " + word) <= 35 and not line2:
                line1 = (line1 + " " + word).strip()
            else:
                line2 = (line2 + " " + word).strip()
        
        if len(line2) > 35:
            line2 = line2[:32] + "..."
    else:
        line1 = product_name
        line2 = ""
    
    # Format dates
    try:
        if pd.notna(date_str) and date_str and str(date_str).strip():
            date_obj = pd.to_datetime(date_str)
            formatted_date = date_obj.strftime('%m/%d/%Y')
        else:
            formatted_date = datetime.now().strftime('%m/%d/%Y')
    except Exception:
        formatted_date = str(date_str) if date_str else datetime.now().strftime('%m/%d/%Y')
    
    # Handle sell by date
    try:
        if pd.notna(sell_by) and sell_by and str(sell_by).strip():
            sell_by_obj = pd.to_datetime(sell_by)
            formatted_sell_by = sell_by_obj.strftime('%m/%d/%Y')
            show_sell_by = True
        else:
            formatted_sell_by = ""
            show_sell_by = False
    except Exception:
        formatted_sell_by = str(sell_by) if sell_by else ""
        show_sell_by = bool(formatted_sell_by)
    
    # Prepare QR code data - FULL package label, no truncation at all
    qr_data = str(package_label) if pd.notna(package_label) and package_label else ""
    # Clean any extra whitespace but keep the full string
    qr_data = qr_data.strip()
    
    # Format quantity label and make it extra large
    qty_label = f"Qty: {qty}" if label_type == "Case" else f"Bin Qty: {qty}"
    
    # Calculate center position for "Delivered: mm/dd/yy" text 
    center_x = int(width_dots / 2) - 80  # Approximate center for longer text
    
    # QR code positioning - right side
    qr_x = 400  # Right side
    qr_y = 120  # Positioned appropriately
    
    # Package label text positioning - at same level as current category, right aligned
    package_text = str(package_label) if pd.notna(package_label) else ""
    package_y = 155  # Same level as where category currently is
    # Calculate proper right alignment - estimate text width and position accordingly
    # For 16pt font, roughly 10-12 dots per character
    if package_text:
        estimated_text_width = len(package_text) * 10  # Conservative estimate
        package_x = width_dots - estimated_text_width - 10  # Position so text ends at right edge
    else:
        package_x = width_dots - 10
    
    # Start building ZPL
    current_y = top_margin
    zpl = f"""^XA
^CF0,{large_font}
^FO{left_margin},{current_y}^FD{line1}^FS"""
    
    current_y += 35
    
    # Add second line of product name if needed
    if line2:
        zpl += f"""
^FO{left_margin},{current_y}^FD{line2}^FS"""
        current_y += 35
    
    # Batch number (left side)
    zpl += f"""
^CF0,{medium_font}
^FO{left_margin},{current_y}^FDBatch: {batch_no}^FS"""
    current_y += 30
    
    # Qty - make it EXTRA LARGE
    zpl += f"""
^CF0,{extra_large_font}
^FO{left_margin},{current_y}^FD{qty_label}^FS"""
    current_y += 40
    
    # Category (moved up, large font like product name)
    category_text = str(category) if pd.notna(category) else ""
    if category_text:
        zpl += f"""
^CF0,{large_font}
^FO{left_margin},{current_y}^FD{category_text}^FS"""
    current_y += 35
    
    # Pkg Qty (moved down)
    zpl += f"""
^CF0,{medium_font}
^FO{left_margin},{current_y}^FDPkg Qty {pkg_qty}^FS"""
    current_y += 25
    
    # Invoice | Sales Order (moved down)
    invoice_text = f"{invoice_no} | {sales_order}" if invoice_no and sales_order else "Invoice | Sales Order"
    zpl += f"""
^CF0,{medium_font}
^FO{left_margin},{current_y}^FD{invoice_text}^FS"""
    
    # "Delivered: mm/dd/yy" - move to middle of label (vertically centered)
    delivered_text = f"Delivered: {formatted_date}"
    delivered_y = int(height_dots / 2) - 10  # Middle of label
    zpl += f"""
^CF0,{large_font}
^FO{center_x},{delivered_y}^FD{delivered_text}^FS"""
    
    # Sell By (only if there's a sell by date) - position below delivered
    if show_sell_by:
        sell_by_y = delivered_y + 35
        zpl += f"""
^CF0,{medium_font}
^FO{left_margin},{sell_by_y}^FDSell By: {formatted_sell_by}^FS"""
    
    # QR code in right side
    zpl += f"""
^FO{qr_x},{qr_y}^BQN,2,4^FD{qr_data}^FS"""
    
    # Package label text - properly right aligned, no missing characters
    if package_text:
        zpl += f"""
^CF0,{small_font}
^FO{package_x},{package_y}^FD{package_text}^FS"""
    
    zpl += """
^XZ"""
    
    return zpl

def send_zpl_to_printer(zpl_data: str, printer_ip: str, 
                       printer_port: int = 9100) -> Tuple[bool, str]:
    """
    Send ZPL data to Zebra printer via network.
    
    Args:
        zpl_data: ZPL code string
        printer_ip: Printer IP address
        printer_port: Printer port (default 9100)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if not printer_ip:
            return False, "No printer IP address provided"
            
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((printer_ip, printer_port))
        sock.send(zpl_data.encode('utf-8'))
        sock.close()
        return True, "Label sent successfully"
        
    except socket.timeout:
        return False, "Connection timeout - check printer IP and network"
    except ConnectionRefusedError:
        return False, "Connection refused - check printer is on and IP is correct"
    except Exception as e:
        return False, f"Network error: {str(e)}"

def generate_labels_for_dataset(df: pd.DataFrame, label_type: str, 
                               label_width: float, label_height: float, dpi: int) -> List[str]:
    """
    Generate ZPL labels for entire dataset with flexible sizing.
    
    Args:
        df: DataFrame with label data
        label_type: "Case Labels" or "Bin Labels"
        label_width: Label width in inches
        label_height: Label height in inches
        dpi: Printer DPI
        
    Returns:
        List of ZPL code strings
    """
    labels = []
    
    for _, row in df.iterrows():
        # Determine quantity needed and type
        if label_type == "Case Labels":
            qty_needed = int(row.get('Case Labels Needed', 0))
            qty_value = row.get('Case Quantity', '')
            zpl_type = "Case"
        else:  # Bin Labels
            qty_needed = int(row.get('Bin Labels Needed', 0))
            qty_value = row.get('Bin Quantity', '')
            zpl_type = "Bin"
        
        # Generate the required number of labels
        for _ in range(qty_needed):
            zpl = generate_label_zpl(
                product_name=row.get('Product Name', ''),
                batch_no=row.get('Batch No', ''),
                qty=qty_value,
                pkg_qty=row.get('Package Quantity', ''),
                date_str=row.get('Delivery Date', ''),
                package_label=row.get('Package Label', ''),
                sell_by=row.get('Sell by', ''),
                invoice_no=row.get('Invoice No', ''),
                sales_order=row.get('Sales Order Number', ''),
                category=row.get('Category', ''),
                label_type=zpl_type,
                label_width=label_width,
                label_height=label_height,
                dpi=dpi
            )
            labels.append(zpl)
    
    return labels

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

# Process Data Button
if st.sidebar.button("🚀 Process Data", type="primary", 
                    disabled=not (sales_order_file and products_file and packages_file)):
    with st.spinner("Processing your data..."):
        # Load CSV files
        sales_order_df = load_sales_order_csv(sales_order_file)
        products_df = load_standard_csv(products_file, "Products")
        packages_df = load_standard_csv(packages_file, "Packages")
        
        # Check if all files loaded successfully
        if sales_order_df is None or products_df is None or packages_df is None:
            st.error("❌ Failed to load one or more CSV files")
            st.stop()
        
        # Display file information
        file_info = (f"Sales Orders: {len(sales_order_df):,} rows | "
                    f"Products: {len(products_df):,} rows | "
                    f"Packages: {len(packages_df):,} rows")
        st.success("✅ Files loaded successfully!")
        st.info(file_info)
        
        # Process and merge data
        processed_data = merge_data_sources(sales_order_df, products_df, packages_df)
        
        if processed_data is not None:
            st.session_state.processed_data = processed_data
            st.success(f"✅ Successfully processed {len(processed_data):,} records")

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

if st.session_state.processed_data is not None:
    processed_df = st.session_state.processed_data
    
    # Create tabs for organization
    tab1, tab2 = st.tabs(["🎯 Label Data Generator", "📊 Data Overview"])
    
    with tab1:
        st.header("🎯 Create Custom Label Data")
        
        # =============================================================================
        # FILTERING SECTION WITH CASCADING LOGIC
        # =============================================================================
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            customers = sorted(processed_df['Customer'].dropna().unique().tolist())
            selected_customers = st.multiselect("Select Customers", customers)
            
        with col2:
            # Cascade: filter orders by selected customers
            if selected_customers:
                filtered_orders = processed_df[
                    processed_df['Customer'].isin(selected_customers)
                ]['Sales Order Number'].dropna().unique()
            else:
                filtered_orders = processed_df['Sales Order Number'].dropna().unique()
            orders = sorted(filtered_orders.tolist())
            selected_orders = st.multiselect("Select Sales Orders", orders)
            
        with col3:
            # Cascade: filter invoices by customers or orders
            if selected_customers:
                filtered_invoices = processed_df[
                    processed_df['Customer'].isin(selected_customers)
                ]['Invoice No'].dropna().unique()
            elif selected_orders:
                filtered_invoices = processed_df[
                    processed_df['Sales Order Number'].isin(selected_orders)
                ]['Invoice No'].dropna().unique()
            else:
                filtered_invoices = processed_df['Invoice No'].dropna().unique()
            invoices = sorted(filtered_invoices.tolist())
            selected_invoices = st.multiselect("Select Invoices", invoices)
        
        with col4:
            # Cascade: filter dates by previous selections
            if selected_customers:
                filtered_dates = processed_df[
                    processed_df['Customer'].isin(selected_customers)
                ]['Delivery Date'].dropna().unique()
            elif selected_orders:
                filtered_dates = processed_df[
                    processed_df['Sales Order Number'].isin(selected_orders)
                ]['Delivery Date'].dropna().unique()
            elif selected_invoices:
                filtered_dates = processed_df[
                    processed_df['Invoice No'].isin(selected_invoices)
                ]['Delivery Date'].dropna().unique()
            else:
                filtered_dates = processed_df['Delivery Date'].dropna().unique()
            delivery_dates = sorted(filtered_dates.tolist())
            selected_dates = st.multiselect("Select Delivery Dates", delivery_dates)
        
        # Apply filters to dataset
        filtered_df = processed_df.copy()
        
        if selected_customers:
            filtered_df = filtered_df[filtered_df['Customer'].isin(selected_customers)]
            
        if selected_orders:
            filtered_df = filtered_df[filtered_df['Sales Order Number'].isin(selected_orders)]
            
        if selected_invoices:
            filtered_df = filtered_df[filtered_df['Invoice No'].isin(selected_invoices)]
            
        if selected_dates:
            filtered_df = filtered_df[filtered_df['Delivery Date'].isin(selected_dates)]
        
        # =============================================================================
        # DATA DISPLAY AND ROW SELECTION
        # =============================================================================
        
        st.subheader(f"🏷️ Label Data Results ({len(filtered_df):,} records)")
        
        # Row selection
        st.subheader("Row Selection")
        select_all = st.checkbox("Select all rows", value=True)
        
        if not select_all:
            selected_indices = st.multiselect(
                "Select specific rows to include:",
                options=filtered_df.index.tolist(),
                default=filtered_df.index.tolist(),
                format_func=lambda x: f"Row {x}: {filtered_df.loc[x, 'Product Name'] if 'Product Name' in filtered_df.columns else 'N/A'}"
            )
            display_data = filtered_df.loc[selected_indices]
        else:
            display_data = filtered_df
        
        # Display the data
        st.dataframe(display_data, width='stretch', height=400)
        
        # =============================================================================
        # EXPORT AND PRINTING SECTION
        # =============================================================================
        
        # Export section
        st.header("💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_buffer = io.StringIO()
            display_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"label_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Summary stats
            st.write("**Summary:**")
            st.write(f"Total rows: {len(display_data):,}")
            if 'Package Quantity' in display_data.columns:
                total_packages = display_data['Package Quantity'].sum()
                st.write(f"Total packages: {total_packages:,}")
            if 'Case Labels Needed' in display_data.columns:
                total_case_labels = display_data['Case Labels Needed'].sum()
                st.write(f"Total case labels needed: {total_case_labels:,}")
            if 'Bin Labels Needed' in display_data.columns:
                total_bin_labels = display_data['Bin Labels Needed'].sum()
                st.write(f"Total bin labels needed: {total_bin_labels:,}")
            if 'Customer' in display_data.columns:
                unique_customers = display_data['Customer'].nunique()
                st.write(f"Unique customers: {unique_customers}")
        
        # =============================================================================
        # ZEBRA LABEL PRINTING SECTION
        # =============================================================================
        
        if QR_AVAILABLE:
            st.header("🖨️ Zebra Label Printing")
            
            # Printer settings - keep it simple for now
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                label_type = st.selectbox(
                    "Label Type",
                    ["Case Labels", "Bin Labels"],
                    help="Choose whether to generate case or bin labels"
                )
            
            with col2:
                printer_connection = st.selectbox(
                    "Connection",
                    ["Network (IP)", "USB/Local"],
                    help="How is your Zebra printer connected?"
                )
            
            with col3:
                if printer_connection == "Network (IP)":
                    printer_ip = st.text_input(
                        "Printer IP Address",
                        placeholder="192.168.1.100",
                        help="Enter the IP address of your Zebra printer"
                    )
                else:
                    st.info("USB printing requires additional setup")
                    printer_ip = None
            
            # Fixed settings for current setup
            label_width = 1.75
            label_height = 0.875
            selected_dpi = 300
            
            # Calculate labels for selected type
            if label_type == "Case Labels":
                total_labels_needed = display_data['Case Labels Needed'].sum()
                label_column = 'Case Labels Needed'
            else:
                total_labels_needed = display_data['Bin Labels Needed'].sum()
                label_column = 'Bin Labels Needed'
            
            # Show label preview and printing options
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader(f"📋 {label_type} Preview")
                
                # Show breakdown of labels by product
                label_breakdown = display_data[display_data[label_column] > 0].copy()
                if len(label_breakdown) > 0:
                    st.write(f"**{len(label_breakdown)} products** will generate **{int(total_labels_needed)} labels**")
                    
                    # Show preview table
                    preview_df = label_breakdown[['Product Name', 'Customer', label_column]].head(10)
                    st.dataframe(preview_df, width='stretch')
                    
                    if len(label_breakdown) > 10:
                        st.write(f"... and {len(label_breakdown) - 10} more products")
                else:
                    st.warning(f"No {label_type.lower()} needed for selected data")
            
            with col2:
                st.subheader("🎯 Print Options")
                
                if total_labels_needed > 0:
                    st.metric("Total Labels", f"{int(total_labels_needed)}")
                    
                    # Generate ZPL preview
                    if st.button("👀 Preview ZPL", help="Generate ZPL code preview"):
                        if len(display_data) > 0:
                            sample_row = display_data.iloc[0]
                            qty_value = (sample_row.get('Case Quantity', '') if label_type == "Case Labels" 
                                       else sample_row.get('Bin Quantity', ''))
                            zpl_type = "Case" if label_type == "Case Labels" else "Bin"
                            
                            sample_zpl = generate_label_zpl(
                                product_name=sample_row.get('Product Name', ''),
                                batch_no=sample_row.get('Batch No', ''),
                                qty=qty_value,
                                pkg_qty=sample_row.get('Package Quantity', ''),
                                date_str=sample_row.get('Delivery Date', ''),
                                package_label=sample_row.get('Package Label', ''),
                                sell_by=sample_row.get('Sell by', ''),
                                invoice_no=sample_row.get('Invoice No', ''),
                                sales_order=sample_row.get('Sales Order Number', ''),
                                category=sample_row.get('Category', ''),
                                label_type=zpl_type,
                                label_width=label_width,
                                label_height=label_height,
                                dpi=selected_dpi
                            )
                            
                            st.code(sample_zpl, language="text")
                            
                            # Show calculated dimensions for reference
                            width_dots = int(label_width * selected_dpi)
                            height_dots = int(label_height * selected_dpi)
                            st.info(f"Label: {label_width}\" × {label_height}\" = {width_dots} × {height_dots} dots at {selected_dpi} DPI")
                    
                    # Print buttons
                    if printer_connection == "Network (IP)" and printer_ip:
                        if st.button(f"🖨️ Print All {label_type}", type="primary"):
                            with st.spinner(f"Generating and printing {int(total_labels_needed)} labels..."):
                                success_count = 0
                                error_messages = []
                                
                                labels = generate_labels_for_dataset(display_data, label_type, label_width, label_height, selected_dpi)
                                
                                for zpl in labels:
                                    success, message = send_zpl_to_printer(zpl, printer_ip)
                                    if success:
                                        success_count += 1
                                    else:
                                        error_messages.append(message)
                                        break  # Stop on first error
                                
                                if error_messages:
                                    st.error(f"❌ Printing failed: {error_messages[0]}")
                                    if success_count > 0:
                                        st.info(f"✅ {success_count} labels printed before error")
                                else:
                                    st.success(f"✅ Successfully printed {success_count} labels!")
                    
                    # Download ZPL file option
                    if st.button("📥 Download ZPL File"):
                        labels = generate_labels_for_dataset(display_data, label_type, label_width, label_height, selected_dpi)
                        zpl_content = "\n".join(labels)
                        
                        # Create descriptive filename with label size, customer, invoice, and sales order
                        filename_parts = []
                        filename_parts.append(f"{label_width}x{label_height}")
                        
                        # Add customer info if single customer
                        customers = display_data['Customer'].dropna().unique()
                        if len(customers) == 1:
                            customer_clean = str(customers[0]).replace(' ', '_').replace('/', '_')
                            filename_parts.append(customer_clean)
                        
                        # Add invoice if single invoice
                        invoices = display_data['Invoice No'].dropna().unique()
                        if len(invoices) == 1:
                            invoice_clean = str(invoices[0]).replace(' ', '_').replace('/', '_')
                            filename_parts.append(f"INV-{invoice_clean}")
                        
                        # Add sales order if single sales order
                        sales_orders = display_data['Sales Order Number'].dropna().unique()
                        if len(sales_orders) == 1:
                            so_clean = str(sales_orders[0]).replace(' ', '_').replace('/', '_')
                            filename_parts.append(f"SO-{so_clean}")
                        
                        # Add label type and timestamp
                        label_type_clean = label_type.lower().replace(' ', '_')
                        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        
                        filename = f"{label_type_clean}_{'_'.join(filename_parts)}_{timestamp}.zpl"
                        
                        st.download_button(
                            label="📄 Download ZPL File",
                            data=zpl_content,
                            file_name=filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                else:
                    st.warning("No labels to print")
                    st.info("Adjust filters or check data")
        else:
            st.error("🚫 QR code libraries not available. Install with: pip install qrcode[pil]")
    
    # =============================================================================
    # DATA OVERVIEW TAB
    # =============================================================================
    
    with tab2:
        st.header("📊 Data Overview")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🛒 Total Sales Orders", len(processed_df['Sales Order Number'].dropna().unique()))
        with col2:
            st.metric("👥 Unique Customers", len(processed_df['Customer'].dropna().unique()))
        with col3:
            st.metric("📦 Total Items", len(processed_df))
        with col4:
            st.metric("🏷️ Categories", len(processed_df['Category'].dropna().unique()))
        with col5:
            # Show label calculation coverage
            case_labels_coverage = processed_df['Case Labels Needed'].gt(0).sum()
            bin_labels_coverage = processed_df['Bin Labels Needed'].gt(0).sum()
            max_coverage = max(case_labels_coverage, bin_labels_coverage)
            coverage_pct = (max_coverage / len(processed_df) * 100) if len(processed_df) > 0 else 0
            st.metric("📊 Label Coverage", f"{coverage_pct:.0f}%")
        
        # Category breakdown
        st.subheader("📈 Category Breakdown")
        category_counts = processed_df['Category'].value_counts()
        st.bar_chart(category_counts)
        
        # Customer breakdown
        st.subheader("👥 Customer Breakdown")
        customer_counts = processed_df['Customer'].value_counts().head(10)
        st.bar_chart(customer_counts)
        
        # Label analysis
        st.subheader("🏷️ Label Requirements Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Case labels analysis
            case_labels_data = processed_df[processed_df['Case Labels Needed'] > 0]
            st.write(f"**Items requiring case labels:** {len(case_labels_data):,} of {len(processed_df):,}")
            
            if len(case_labels_data) > 0:
                total_case_labels = case_labels_data['Case Labels Needed'].sum()
                avg_case_labels = case_labels_data['Case Labels Needed'].mean()
                st.write(f"**Total case labels needed:** {total_case_labels:,}")
                st.write(f"**Average per item:** {avg_case_labels:.2f}")
        
        with col2:
            # Bin labels analysis
            bin_labels_data = processed_df[processed_df['Bin Labels Needed'] > 0]
            st.write(f"**Items requiring bin labels:** {len(bin_labels_data):,} of {len(processed_df):,}")
            
            if len(bin_labels_data) > 0:
                total_bin_labels = bin_labels_data['Bin Labels Needed'].sum()
                avg_bin_labels = bin_labels_data['Bin Labels Needed'].mean()
                st.write(f"**Total bin labels needed:** {total_bin_labels:,}")
                st.write(f"**Average per item:** {avg_bin_labels:.2f}")
        
        # Full dataset view
        st.subheader("🔍 Full Dataset")
        st.dataframe(processed_df, width='stretch')

# =============================================================================
# WELCOME SCREEN
# =============================================================================

else:
    # Welcome screen when no data is loaded
    if not sales_order_file and not products_file and not packages_file:
        st.info("👈 Upload the required CSV files in the sidebar to get started")
        
        # Show helpful information
        with st.expander("ℹ️ How it Works", expanded=True):
            st.markdown("""
            **📋 Upload** → **🔄 Process** → **🎯 Filter** → **🖨️ Print**
            
            **Haven Cannabis Label Data Processor v1.0** processes your CSV files to create label printing data with calculated quantities and direct Zebra printer support.
            
            **Key Features:**
            - 🔗 Links Sales Orders with Products and Packages
            - 📊 Calculates Case Labels Needed = ⌈Package Quantity ÷ Case Quantity⌉
            - 📊 Calculates Bin Labels Needed = ⌈Package Quantity ÷ Bin Quantity⌉
            - 📅 Formats delivery dates as mm-dd-yy
            - 🎯 Cascading filters by customer, order, invoice, and date
            - 📋 Row selection for precise control
            - 🖨️ Direct Zebra printer support (ZD410, ZD621 at 300 DPI)
            - 📄 ZPL generation for 1.75" x 0.875" labels
            - 🔧 Smart layout with auto-wrapping and positioning
            - 🔗 QR codes for package tracking
            - 📊 Data overview and analytics
            - 📥 CSV and ZPL export functionality
            """)
        
        with st.expander("🖨️ Zebra Printer Setup"):
            st.markdown("""
            **Supported Printers:** Zebra ZD410, ZD621 (300 DPI)
            
            **Current Label Specifications:**
            - Size: 1.75" wide × 0.875" high
            - Resolution: 300 DPI
            - Formats: Case Labels and Bin Labels
            
            **Connection Options:**
            - **Network (Recommended):** Enter printer IP address
            - **USB:** Requires additional driver setup
            
            **Label Content:**
            - Product Name (auto-wrapping for long names)
            - Batch Number
            - Quantity (Case Qty or Bin Qty)
            - Package Quantity  
            - Date (formatted as M/D/YYYY)
            - Sell By Date
            - QR Code with Package Label data
            
            **Features:**
            - **Smart layout:** Text positioning optimized for 1.75" × 0.875" labels
            - **Dynamic scaling:** Fonts and spacing calculated for 300 DPI
            - **QR positioning:** Automatically placed in optimal location
            - **Product name wrapping:** Long names split across two lines
            """)
            
        with st.expander("📁 CSV File Requirements"):
            st.markdown("""
            **Sales Order CSV:** *(Required)*
            - Headers start on line 4 (first 3 lines skipped automatically)
            - Required columns: Product Id, Product, Category, Package Batch Number, Package Label, Quantity, Customer, Invoice Numbers, Metrc Manifest Number, Delivery Date, Order Number
            
            **Products CSV:** *(Required)*  
            - Required columns: ID, Units Per Case, Bin Quantity (Retail)
            - Used for calculating label quantities
            
            **Package List CSV:** *(Required)*
            - Required columns: Package Label, Sell By
            - Links package information to sales orders
            
            **Data Linking:**
            - Product Id (Sales Order) ↔ ID (Products)
            - Package Label (Sales Order) ↔ Package Label (Packages)
            """)
    
    elif sales_order_file and products_file and packages_file:
        st.info("👈 Click the 'Process Data' button in the sidebar to analyze your files")
        st.info("🏷️ All files uploaded - ready to process label data and print to Zebra printers")
    
    else:
        missing_files = []
        if not sales_order_file:
            missing_files.append("Sales Order")
        if not products_file:
            missing_files.append("Products")
        if not packages_file:
            missing_files.append("Package List")
        
        st.warning(f"📁 Please upload the {' and '.join(missing_files)} CSV file(s) to continue")