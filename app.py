"""
Haven Cannabis Label Data Processor v1.0 - Complete Rewrite
============================================================

A Streamlit application for processing CSV files and generating Zebra printer labels.
Supports direct printing to Zebra ZD410/ZD621 printers via network connection.

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

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Haven Cannabis - Label Data Processor",
    page_icon="🏷️",
    layout="wide"
)

# App header
st.title("🏷️ Label Data Processor v1.0")
st.markdown("**Haven Cannabis** | Process CSV files for label printing with calculated quantities")

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = [
        'processed_data',
        'sales_order_data', 
        'products_data',
        'packages_data',
        'case_labels_data',    # NEW: Label-ready case dataset
        'bin_labels_data'      # NEW: Label-ready bin dataset
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

initialize_session_state()

# =============================================================================
# LABEL-READY DATASET GENERATION
# =============================================================================

def create_case_labels_dataset(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataset where each row represents exactly one case label to print.
    Handles partial cases when Package Qty doesn't divide evenly by Case Qty.
    """
    case_labels_data = []
    
    for _, row in merged_df.iterrows():
        package_qty = safe_numeric(row.get('Package Quantity', 0))
        case_qty = safe_numeric(row.get('Case Quantity', 0))
        
        if package_qty > 0 and case_qty > 0:
            # Calculate individual case quantities (same logic as bins)
            remaining = package_qty
            label_number = 1
            
            while remaining > 0:
                if remaining >= case_qty:
                    # Full case
                    actual_case_qty = case_qty
                    remaining -= case_qty
                else:
                    # Partial case (remainder)
                    actual_case_qty = remaining
                    remaining = 0
                
                # Create a new row for this specific label
                label_row = row.copy()
                label_row['Actual Case Qty'] = actual_case_qty
                label_row['Label Number'] = label_number
                label_row['Label Type'] = 'Case'
                
                case_labels_data.append(label_row)
                label_number += 1
    
    return pd.DataFrame(case_labels_data) if case_labels_data else pd.DataFrame()

def create_bin_labels_dataset(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataset where each row represents exactly one bin label to print.
    Handles partial bins when Package Qty doesn't divide evenly by Bin Qty.
    """
    bin_labels_data = []
    
    for _, row in merged_df.iterrows():
        package_qty = safe_numeric(row.get('Package Quantity', 0))
        bin_qty = safe_numeric(row.get('Bin Quantity', 0))
        product_name = row.get('Product Name', 'Unknown Product')
        
        # DEBUG: Show what we're processing
        st.write(f"🔍 DATASET DEBUG: {product_name} - Package: {package_qty}, Bin: {bin_qty}")
        
        if package_qty > 0 and bin_qty > 0:
            # Calculate individual bin quantities
            remaining = package_qty
            label_number = 1
            
            while remaining > 0:
                if remaining >= bin_qty:
                    # Full bin
                    actual_bin_qty = bin_qty
                    remaining -= bin_qty
                else:
                    # Partial bin (remainder)
                    actual_bin_qty = remaining
                    remaining = 0
                
                # DEBUG: Show what we're creating
                st.write(f"🔍 DATASET DEBUG: Creating label {label_number} with Actual Bin Qty: {actual_bin_qty}")
                
                # Create a new row for this specific label
                label_row = row.copy()
                label_row['Actual Bin Qty'] = actual_bin_qty
                label_row['Label Number'] = label_number
                label_row['Label Type'] = 'Bin'
                
                bin_labels_data.append(label_row)
                label_number += 1
    
    result_df = pd.DataFrame(bin_labels_data) if bin_labels_data else pd.DataFrame()
    st.write(f"🔍 DATASET DEBUG: Created bin dataset with {len(result_df)} total rows")
    
    return result_df

# =============================================================================
# ADVANCED LABEL QUANTITY CALCULATIONS (DEPRECATED - KEPT FOR REFERENCE)
# =============================================================================

def calculate_individual_bin_quantities(package_qty: float, bin_qty: float) -> List[float]:
    """
    Calculate the actual quantity that should appear on each individual bin label.
    
    Examples:
    - Package=100, Bin=20 → [20, 20, 20, 20, 20] (5 full bins)
    - Package=50, Bin=30 → [30, 20] (1 full bin + 1 partial bin)
    
    Args:
        package_qty: Total package quantity
        bin_qty: Capacity of each bin
        
    Returns:
        List of quantities for each label
    """
    # DEBUG: Show inputs
    print(f"🔍 CALC DEBUG: package_qty={package_qty}, bin_qty={bin_qty}")
    
    if package_qty <= 0 or bin_qty <= 0:
        print(f"🔍 CALC DEBUG: Invalid inputs, returning empty list")
        return []
    
    quantities = []
    remaining = package_qty
    iteration = 0
    
    while remaining > 0:
        iteration += 1
        print(f"🔍 CALC DEBUG: Iteration {iteration}, remaining={remaining}")
        
        if remaining >= bin_qty:
            # Full bin
            quantities.append(bin_qty)
            remaining -= bin_qty
            print(f"🔍 CALC DEBUG: Added full bin {bin_qty}, remaining now {remaining}")
        else:
            # Partial bin (remainder)
            quantities.append(remaining)
            print(f"🔍 CALC DEBUG: Added partial bin {remaining}")
            remaining = 0
    
    print(f"🔍 CALC DEBUG: Final result: {quantities}")
    return quantities

# =============================================================================
# UTILITY FUNCTIONS FOR ROBUST NUMERIC HANDLING
# =============================================================================

def safe_numeric(value, default=0):
    """
    Convert any value to numeric, handling strings, NaN, None, etc.
    Always returns a number (int or float).
    """
    if pd.isna(value) or value is None or value == '':
        return default
    try:
        # Handle string numbers that might have leading/trailing whitespace
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                return default
        # Convert to float first, then to int if it's a whole number
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

# =============================================================================
# FILE UPLOAD INTERFACE
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
    Uses dtype=str to preserve package labels like "1A40603000067FG000030637".
    """
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

def calculate_label_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Case Labels Needed and Bin Labels Needed with robust numeric handling.
    """
    # Case Labels Needed calculation
    if 'Package Quantity' in df.columns and 'Case Quantity' in df.columns:
        case_labels = []
        for _, row in df.iterrows():
            pkg_qty = safe_numeric(row.get('Package Quantity', 0))
            case_qty = safe_numeric(row.get('Case Quantity', 0))
            
            if pkg_qty > 0 and case_qty > 0:
                case_labels.append(math.ceil(pkg_qty / case_qty))
            else:
                case_labels.append(0)
        df['Case Labels Needed'] = case_labels
    else:
        df['Case Labels Needed'] = 0
    
    # Bin Labels Needed calculation
    if 'Package Quantity' in df.columns and 'Bin Quantity' in df.columns:
        bin_labels = []
        for _, row in df.iterrows():
            pkg_qty = safe_numeric(row.get('Package Quantity', 0))
            bin_qty = safe_numeric(row.get('Bin Quantity', 0))
            
            if pkg_qty > 0 and bin_qty > 0:
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
    """
    Merge the three data sources and create label-ready datasets.
    
    NEW APPROACH: Creates separate datasets for case and bin labels where each row = 1 label
    """
    try:
        # Store raw data
        st.session_state.sales_order_data = sales_order_df
        st.session_state.products_data = products_df
        st.session_state.packages_data = packages_df
        
        # Merge operations (same as before)
        merged_df = sales_order_df.merge(
            products_df, 
            left_on='Product Id', 
            right_on='ID', 
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
        
        # Column mapping (same as before)
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
        
        # Create base dataset
        base_data = pd.DataFrame()
        for original_col, new_col in column_mapping.items():
            if original_col in final_df.columns:
                base_data[new_col] = final_df[original_col]
            else:
                base_data[new_col] = None
        
        # Format delivery dates
        if 'Delivery Date' in base_data.columns:
            base_data['Delivery Date'] = format_delivery_date(base_data['Delivery Date'])
        
        # NEW APPROACH: Create label-ready datasets
        st.info("🔄 Creating label-ready datasets...")
        
        # Create case labels dataset (each row = 1 case label)
        case_labels_df = create_case_labels_dataset(base_data)
        st.session_state.case_labels_data = case_labels_df
        
        # Create bin labels dataset (each row = 1 bin label)  
        bin_labels_df = create_bin_labels_dataset(base_data)
        st.session_state.bin_labels_data = bin_labels_df
        
        # Show summary
        case_count = len(case_labels_df) if not case_labels_df.empty else 0
        bin_count = len(bin_labels_df) if not bin_labels_df.empty else 0
        
        st.success(f"✅ Created {case_count} case labels and {bin_count} bin labels")
        
        # Return the base data for overview purposes (add summary columns for display)
        base_data['Case Labels Needed'] = base_data.apply(
            lambda row: len(create_case_labels_dataset(pd.DataFrame([row]))) if safe_numeric(row.get('Package Quantity', 0)) > 0 else 0, 
            axis=1
        )
        base_data['Bin Labels Needed'] = base_data.apply(
            lambda row: len(create_bin_labels_dataset(pd.DataFrame([row]))) if safe_numeric(row.get('Package Quantity', 0)) > 0 else 0, 
            axis=1
        )
        
        # Reorder columns
        desired_order = [
            'Product Name', 'Category', 'Batch No', 'Package Label', 
            'Package Quantity', 'Case Quantity', 'Bin Quantity', 
            'Case Labels Needed', 'Bin Labels Needed',
            'Customer', 'Invoice No', 'Sales Order Number', 'METRC Manifest', 
            'Delivery Date', 'Sell by'
        ]
        
        for col in desired_order:
            if col not in base_data.columns:
                base_data[col] = None
        
        return base_data[desired_order]
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

# =============================================================================
# ZEBRA PRINTING FUNCTIONS  
# =============================================================================

def sanitize_qr_data(package_label) -> str:
    """
    Preserve complete package label for QR code.
    FIXED: The truncation issue was ZPL format, not data preservation.
    """
    if pd.isna(package_label) or package_label is None:
        return ""
    
    # Convert to string and strip whitespace only
    qr_data = str(package_label).strip()
    return qr_data

def generate_bin_label_zpl(product_name: str, batch_no: str, actual_bin_qty: float,
                          pkg_qty: str, date_str: str, package_label: str, 
                          sell_by: str, invoice_no: str, metrc_manifest: str, category: str,
                          label_width: float = 1.75, label_height: float = 0.875, 
                          dpi: int = 300) -> str:
    """
    Generate ZPL code specifically for BIN labels with quantity baked in.
    
    CORRECT APPROACH: Each label is completely separate with its specific quantity
    """
    # Calculate dimensions
    width_dots = int(label_width * dpi)
    height_dots = int(label_height * dpi)
    
    # Font sizes
    fonts = {
        'extra_large': 32,  # Quantity
        'large': 28,        # Product name, Category  
        'medium': 20,       # Batch, Delivered
        'small': 16,        # Pkg Qty line
        'small_plus': 18    # Bottom line
    }
    
    # Layout constants
    left_margin = 30
    top_margin = 20
    
    # Preserve complete QR data
    qr_data = sanitize_qr_data(package_label)
    
    # Handle product name wrapping
    product_name = str(product_name) if pd.notna(product_name) else ""
    product_lines = []
    
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
    
    # Handle sell by date
    sell_by_text = ""
    if pd.notna(sell_by) and sell_by and str(sell_by).strip():
        try:
            sell_by_obj = pd.to_datetime(sell_by)
            sell_by_text = f"Sell By: {sell_by_obj.strftime('%m/%d/%Y')}"
        except Exception:
            sell_by_text = f"Sell By: {str(sell_by)}"
    
    # Layout positioning
    qr_size = 5
    qr_x = width_dots - 140
    qr_y = height_dots - 160  # Positioned with clearance
    
    # Bottom area
    pkg_qty_y = height_dots - 35
    combined_line_y = height_dots - 15
    
    # Create combined bottom line text
    combined_parts = []
    if invoice_no:
        combined_parts.append(str(invoice_no))
    if metrc_manifest:
        combined_parts.append(str(metrc_manifest))
    if qr_data:
        display_package = qr_data[:25] + "..." if len(qr_data) > 25 else qr_data
        combined_parts.append(display_package)
    
    combined_text = " | ".join(combined_parts)
    
    # Build ZPL - BAKED-IN QUANTITY APPROACH
    current_y = top_margin
    zpl_lines = ["^XA", f"^CF0,{fonts['large']}"]
    
    # Product name
    for line in product_lines:
        zpl_lines.append(f"^FO{left_margin},{current_y}^FD{line}^FS")
        current_y += 35
    
    # Batch number
    zpl_lines.extend([
        f"^CF0,{fonts['medium']}",
        f"^FO{left_margin},{current_y}^FDBatch: {batch_no}^FS"
    ])
    current_y += 30
    
    # Quantity - BAKED IN, NOT PARAMETER-BASED
    # Format the actual bin quantity directly into the ZPL
    if actual_bin_qty == int(actual_bin_qty):  # If it's a whole number
        qty_display = f"Bin Qty: {int(actual_bin_qty)}"
    else:
        qty_display = f"Bin Qty: {actual_bin_qty}"
        
    zpl_lines.extend([
        f"^CF0,{fonts['extra_large']}",
        f"^FO{left_margin},{current_y}^FD{qty_display}^FS"
    ])
    current_y += 40
    
    # Category
    category_text = str(category) if pd.notna(category) else ""
    if category_text:
        zpl_lines.extend([
            f"^CF0,{fonts['large']}",
            f"^FO{left_margin},{current_y}^FD{category_text}^FS"
        ])
        current_y += 35
    
    # Delivered date
    zpl_lines.extend([
        f"^CF0,{fonts['medium']}",
        f"^FO{left_margin},{current_y}^FDDelivered: {formatted_date}^FS"
    ])
    
    # Sell by date (if available)
    if sell_by_text:
        current_y += 30
        zpl_lines.extend([
            f"^CF0,{fonts['medium']}",
            f"^FO{left_margin},{current_y}^FD{sell_by_text}^FS"
        ])
    
    # QR code with proper format
    if qr_data:
        zpl_lines.append(f"^FO{qr_x},{qr_y}^BQN,2,{qr_size}^FDQA,{qr_data}^FS")
    
    # Pkg Qty line
    if pkg_qty:
        zpl_lines.extend([
            f"^CF0,{fonts['small']}",
            f"^FO{left_margin},{pkg_qty_y}^FDPkg Qty: {pkg_qty}^FS"
        ])
    
    # Combined bottom line
    if combined_text:
        zpl_lines.extend([
            f"^CF0,{fonts['small_plus']}",
            f"^FO{left_margin},{combined_line_y}^FD{combined_text}^FS"
        ])
    
    zpl_lines.append("^XZ")
    return "\n".join(zpl_lines)

def generate_label_zpl(product_name: str, batch_no: str, qty: str, 
                      pkg_qty: str, date_str: str, package_label: str, 
                      sell_by: str, invoice_no: str, metrc_manifest: str, category: str,
                      label_type: str = "Case",
                      label_width: float = 1.75, label_height: float = 0.875, 
                      dpi: int = 300) -> str:
    """
    Generate ZPL code for Zebra printer labels with fixed layout.
    
    FIXED: QR code now uses proper ZPL format with QA, switches to prevent truncation
    UPDATED: Uses METRC Manifest Number instead of Sales Order Number
    """
    # Calculate dimensions
    width_dots = int(label_width * dpi)
    height_dots = int(label_height * dpi)
    
    # Font sizes
    fonts = {
        'extra_large': 32,  # Quantity
        'large': 28,        # Product name, Category  
        'medium': 20,       # Batch, Delivered
        'small': 16,        # Pkg Qty line
        'small_plus': 18    # Bottom line (increased by 2pts)
    }
    
    # Layout constants
    left_margin = 30
    top_margin = 20
    
    # Preserve complete QR data
    qr_data = sanitize_qr_data(package_label)
    
    # Handle product name wrapping
    product_name = str(product_name) if pd.notna(product_name) else ""
    product_lines = []
    
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
    
    # Handle sell by date
    sell_by_text = ""
    if pd.notna(sell_by) and sell_by and str(sell_by).strip():
        try:
            sell_by_obj = pd.to_datetime(sell_by)
            sell_by_text = f"Sell By: {sell_by_obj.strftime('%m/%d/%Y')}"
        except Exception:
            sell_by_text = f"Sell By: {str(sell_by)}"
    
    # Layout positioning
    qr_size = 5
    qr_x = width_dots - 140
    qr_y = height_dots - 160  # Moved up further to prevent overlap with bottom text
    
    # Bottom area
    pkg_qty_y = height_dots - 35
    combined_line_y = height_dots - 15
    
    # Create combined bottom line text: Invoice | METRC Manifest | Package Label
    combined_parts = []
    if invoice_no:
        combined_parts.append(str(invoice_no))
    if metrc_manifest:
        combined_parts.append(str(metrc_manifest))  # Raw METRC number only
    if qr_data:  # Use same qr_data to ensure consistency
        # Truncate if too long for combined line
        display_package = qr_data[:25] + "..." if len(qr_data) > 25 else qr_data
        combined_parts.append(display_package)
    
    combined_text = " | ".join(combined_parts)
    
    # Build ZPL
    current_y = top_margin
    zpl_lines = ["^XA", f"^CF0,{fonts['large']}"]
    
    # Product name
    for line in product_lines:
        zpl_lines.append(f"^FO{left_margin},{current_y}^FD{line}^FS")
        current_y += 35
    
    # Batch number
    zpl_lines.extend([
        f"^CF0,{fonts['medium']}",
        f"^FO{left_margin},{current_y}^FDBatch: {batch_no}^FS"
    ])
    current_y += 30
    
    # Quantity (extra large)
    qty_label = f"Qty: {qty}" if label_type == "Case" else f"Bin Qty: {qty}"
    zpl_lines.extend([
        f"^CF0,{fonts['extra_large']}",
        f"^FO{left_margin},{current_y}^FD{qty_label}^FS"
    ])
    current_y += 40
    
    # Category
    category_text = str(category) if pd.notna(category) else ""
    if category_text:
        zpl_lines.extend([
            f"^CF0,{fonts['large']}",
            f"^FO{left_margin},{current_y}^FD{category_text}^FS"
        ])
        current_y += 35
    
    # Delivered date (under category)
    zpl_lines.extend([
        f"^CF0,{fonts['medium']}",
        f"^FO{left_margin},{current_y}^FDDelivered: {formatted_date}^FS"
    ])
    
    # Sell by date (if available)
    if sell_by_text:
        current_y += 30
        zpl_lines.extend([
            f"^CF0,{fonts['medium']}",
            f"^FO{left_margin},{current_y}^FD{sell_by_text}^FS"
        ])
    
    # QR code (larger, positioned above bottom text) - FIXED with proper ZPL QR switches
    if qr_data:
        # SOLUTION: Add QR switches before data to prevent truncation
        # QA, = Q (error correction level) + A (automatic data input) + comma
        zpl_lines.append(f"^FO{qr_x},{qr_y}^BQN,2,{qr_size}^FDQA,{qr_data}^FS")
    
    # Pkg Qty line
    if pkg_qty:
        zpl_lines.extend([
            f"^CF0,{fonts['small']}",
            f"^FO{left_margin},{pkg_qty_y}^FDPkg Qty: {pkg_qty}^FS"
        ])
    
    # Combined bottom line (larger font)
    if combined_text:
        zpl_lines.extend([
            f"^CF0,{fonts['small_plus']}",
            f"^FO{left_margin},{combined_line_y}^FD{combined_text}^FS"
        ])
    
    zpl_lines.append("^XZ")
    return "\n".join(zpl_lines)

def send_zpl_to_printer(zpl_data: str, printer_ip: str, 
                       printer_port: int = 9100) -> Tuple[bool, str]:
    """Send ZPL data to Zebra printer via network"""
    try:
        if not printer_ip:
            return False, "No printer IP address provided"
            
        socket.inet_aton(printer_ip)  # Validate IP format
        
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

def generate_labels_for_dataset(df: pd.DataFrame, label_type: str, 
                               label_width: float, label_height: float, dpi: int) -> List[str]:
    """Generate ZPL labels for entire dataset"""
    labels = []
    
    for _, row in df.iterrows():
        if label_type == "Case Labels":
            qty_needed = safe_numeric(row.get('Case Labels Needed', 0))
            qty_value = row.get('Case Quantity', '')
            zpl_type = "Case"
        else:
            qty_needed = safe_numeric(row.get('Bin Labels Needed', 0))
            qty_value = row.get('Bin Quantity', '')
            zpl_type = "Bin"
        
        for _ in range(int(qty_needed)):
            zpl = generate_label_zpl(
                product_name=row.get('Product Name', ''),
                batch_no=row.get('Batch No', ''),
                qty=qty_value,
                pkg_qty=row.get('Package Quantity', ''),
                date_str=row.get('Delivery Date', ''),
                package_label=row.get('Package Label', ''),
                sell_by=row.get('Sell by', ''),
                invoice_no=row.get('Invoice No', ''),
                metrc_manifest=row.get('METRC Manifest', ''),  # UPDATED: Use METRC instead of Sales Order
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
        
        if sales_order_df is None or products_df is None or packages_df is None:
            st.error("❌ Failed to load one or more CSV files")
            st.stop()
        
        # Display file info
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
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Label Data Generator", "📊 Data Overview"])
    
    with tab1:
        st.header("🎯 Create Custom Label Data")
        
        # Filtering section
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            customers = sorted(processed_df['Customer'].dropna().unique().tolist())
            selected_customers = st.multiselect("Select Customers", customers)
            
        with col2:
            if selected_customers:
                filtered_orders = processed_df[
                    processed_df['Customer'].isin(selected_customers)
                ]['Sales Order Number'].dropna().unique()
            else:
                filtered_orders = processed_df['Sales Order Number'].dropna().unique()
            orders = sorted(filtered_orders.tolist())
            selected_orders = st.multiselect("Select Sales Orders", orders)
            
        with col3:
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
        
        # Apply filters
        filtered_df = processed_df.copy()
        
        if selected_customers:
            filtered_df = filtered_df[filtered_df['Customer'].isin(selected_customers)]
        if selected_orders:
            filtered_df = filtered_df[filtered_df['Sales Order Number'].isin(selected_orders)]
        if selected_invoices:
            filtered_df = filtered_df[filtered_df['Invoice No'].isin(selected_invoices)]
        if selected_dates:
            filtered_df = filtered_df[filtered_df['Delivery Date'].isin(selected_dates)]
        
        # Data display and row selection
        st.subheader(f"🏷️ Label Data Results ({len(filtered_df):,} records)")
        
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
        
        # Display the data
        if not display_data.empty:
            st.dataframe(display_data, use_container_width=True, height=400)
        else:
            st.warning("No data to display with current selection")
        
        # Export and printing section
        if not display_data.empty:
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
                # Summary stats with SAFE formatting
                st.write("**Summary:**")
                st.write(f"Total rows: {len(display_data):,}")
                
                # Use safe_sum and safe_count_nonzero for all calculations
                total_packages = safe_sum(display_data['Package Quantity'])
                total_case_labels = safe_sum(display_data['Case Labels Needed'])
                total_bin_labels = safe_sum(display_data['Bin Labels Needed'])
                unique_customers = display_data['Customer'].nunique()
                
                # Format numbers safely - ensure they're actually numbers
                if total_packages > 0:
                    st.write(f"Total packages: {float(total_packages):,.1f}")
                if total_case_labels > 0:
                    st.write(f"Total case labels needed: {int(total_case_labels):,}")
                if total_bin_labels > 0:
                    st.write(f"Total bin labels needed: {int(total_bin_labels):,}")
                st.write(f"Unique customers: {unique_customers}")
            
            # Zebra Label Printing Section
            if QR_AVAILABLE:
                st.header("🖨️ Zebra Label Printing")
                
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
                
                # Fixed settings
                label_width = 1.75
                label_height = 0.875
                selected_dpi = 300
                
                # Calculate labels for selected type
                if label_type == "Case Labels":
                    total_labels_needed = safe_sum(display_data['Case Labels Needed'])
                    label_column = 'Case Labels Needed'
                else:
                    total_labels_needed = safe_sum(display_data['Bin Labels Needed'])
                    label_column = 'Bin Labels Needed'
                
                # Show preview and printing options
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader(f"📋 {label_type} Preview")
                    
                    # Filter for products that need labels
                    label_breakdown = display_data[display_data[label_column].apply(lambda x: safe_numeric(x) > 0)].copy()
                    
                    if len(label_breakdown) > 0:
                        st.write(f"**{len(label_breakdown)} products** will generate **{int(total_labels_needed)} labels**")
                        
                        preview_df = label_breakdown[['Product Name', 'Customer', label_column]].head(10)
                        st.dataframe(preview_df, use_container_width=True)
                        
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
                                
                                # Show QR debug info
                                package_label_raw = sample_row.get('Package Label', '')
                                qr_debug_data = sanitize_qr_data(package_label_raw)
                                st.info(f"🔍 QR Debug - Raw: '{package_label_raw}' → QR Data: '{qr_debug_data}' (Length: {len(qr_debug_data)})")
                                
                                sample_zpl = generate_label_zpl(
                                    product_name=sample_row.get('Product Name', ''),
                                    batch_no=sample_row.get('Batch No', ''),
                                    qty=qty_value,
                                    pkg_qty=sample_row.get('Package Quantity', ''),
                                    date_str=sample_row.get('Delivery Date', ''),
                                    package_label=sample_row.get('Package Label', ''),
                                    sell_by=sample_row.get('Sell by', ''),
                                    invoice_no=sample_row.get('Invoice No', ''),
                                    metrc_manifest=sample_row.get('METRC Manifest', ''),  # UPDATED: Use METRC instead of Sales Order
                                    category=sample_row.get('Category', ''),
                                    label_type=zpl_type,
                                    label_width=label_width,
                                    label_height=label_height,
                                    dpi=selected_dpi
                                )
                                
                                st.code(sample_zpl, language="text")
                                
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
                                    
                                    for i, zpl in enumerate(labels, 1):
                                        success, message = send_zpl_to_printer(zpl, printer_ip)
                                        if success:
                                            success_count += 1
                                            if i % 10 == 0:
                                                st.write(f"Printed {i}/{len(labels)} labels...")
                                        else:
                                            error_messages.append(message)
                                            break
                                    
                                    if error_messages:
                                        st.error(f"❌ Printing failed: {error_messages[0]}")
                                        if success_count > 0:
                                            st.info(f"✅ {success_count} labels printed before error")
                                    else:
                                        st.success(f"✅ Successfully printed {success_count} labels!")
                        
                        # Download ZPL file
                        if st.button("📥 Download ZPL File"):
                            labels = generate_labels_for_dataset(display_data, label_type, label_width, label_height, selected_dpi)
                            zpl_content = "\n".join(labels)
                            
                            filename_parts = [f"{label_width}x{label_height}"]
                            
                            customers = display_data['Customer'].dropna().unique()
                            if len(customers) == 1:
                                customer_clean = str(customers[0]).replace(' ', '_').replace('/', '_')
                                filename_parts.append(customer_clean)
                            
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
            else:
                st.error("🚫 QR code libraries not available. Install with: pip install qrcode[pil]")
    
    # Data Overview Tab
    with tab2:
        st.header("📊 Data Overview")
        
        # Summary metrics - ALL SAFE
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
            case_coverage = safe_count_nonzero(processed_df['Case Labels Needed'])
            bin_coverage = safe_count_nonzero(processed_df['Bin Labels Needed'])
            max_coverage = max(case_coverage, bin_coverage)
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
        
        # Label analysis - ALL SAFE
        st.subheader("🏷️ Label Requirements Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            case_items = safe_count_nonzero(processed_df['Case Labels Needed'])
            st.write(f"**Items requiring case labels:** {case_items:,} of {len(processed_df):,}")
            
            if case_items > 0:
                total_case = safe_sum(processed_df['Case Labels Needed'])
                avg_case = total_case / case_items if case_items > 0 else 0
                st.write(f"**Total case labels needed:** {int(total_case):,}")
                st.write(f"**Average per item:** {avg_case:.2f}")
        
        with col2:
            bin_items = safe_count_nonzero(processed_df['Bin Labels Needed'])
            st.write(f"**Items requiring bin labels:** {bin_items:,} of {len(processed_df):,}")
            
            if bin_items > 0:
                total_bin = safe_sum(processed_df['Bin Labels Needed'])
                avg_bin = total_bin / bin_items if bin_items > 0 else 0
                st.write(f"**Total bin labels needed:** {int(total_bin):,}")
                st.write(f"**Average per item:** {avg_bin:.2f}")
        
        # Full dataset view
        st.subheader("🔍 Full Dataset")
        st.dataframe(processed_df, use_container_width=True)


# Welcome screen
else:
    if not sales_order_file and not products_file and not packages_file:
        st.info("👈 Upload the required CSV files in the sidebar to get started")
        
        with st.expander("ℹ️ How it Works", expanded=True):
            st.markdown("""
            **📋 Upload** → **🔄 Process** → **🎯 Filter** → **🖨️ Print**
            
            **Key Features:**
            - 🔗 Links Sales Orders with Products and Packages
            - 📊 Calculates Case/Bin Labels Needed automatically
            - 🎯 Cascading filters by customer, order, invoice, and date
            - 🖨️ Direct Zebra printer support (ZD410, ZD621 at 300 DPI)
            - 🔗 Complete QR codes for package tracking (no truncation)
            - 📥 CSV and ZPL export functionality
            
            **Recent Fixes:**
            - ✅ QR codes preserve complete package label data
            - ✅ Package label text properly positioned
            - ✅ Robust numeric handling for all calculations
            - ✅ No more string formatting errors
            """)
    
    elif sales_order_file and products_file and packages_file:
        st.info("👈 Click the 'Process Data' button in the sidebar to analyze your files")
    
    else:
        missing_files = []
        if not sales_order_file:
            missing_files.append("Sales Order")
        if not products_file:
            missing_files.append("Products")
        if not packages_file:
            missing_files.append("Package List")
        
        st.warning(f"📁 Please upload the {' and '.join(missing_files)} CSV file(s) to continue")