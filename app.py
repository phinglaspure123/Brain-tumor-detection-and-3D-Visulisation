import streamlit as st
import os
import shutil
from detection import img_pred, predict_nii
from visualize import mesh_3d, mesh_3d_affected_area, mesh_3d_tumor, generate_gif, segments_differnt_effects, slice_flair

# styling
with open("main.css") as source_des:
    st.markdown(
        f'<style>{source_des.read()}</style>',
        unsafe_allow_html=True
    ) 

# Create a folder to store uploaded files
UPLOAD_FOLDER = "uploaded_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to load sample data
def load_sample_data(sample_folder):
    flair_path = os.path.join("sample_input_nii", sample_folder, "flair.nii")
    seg_path = os.path.join("sample_input_nii", sample_folder, "seg.nii")
    return flair_path, seg_path

# Function to load sample jpeg data``
def load_sample_data_jpeg(sample_folder):
    jpeg_path=os.path.join("sample_input_jpeg",sample_folder)
    return jpeg_path


# Streamlit web application
def main():
    st.title('Brain Tumor Prediction and 3D Visualization')

    # Sidebar menu for file type selection
    file_type = st.sidebar.radio("Choose File Type", ["JPEG", ".nii"])

    if file_type == "JPEG":
        st.subheader('JPEG File Prediction and Classification')
        st.markdown("---")
        # File upload for JPEG in the sidebar
        st.sidebar.header("Upload JPEG File or Choose Sample Input")
        upload_option = st.sidebar.radio("Select Option", ["Sample Input","Upload Files"])

        if upload_option=="Upload Files":
            uploaded_file = st.sidebar.file_uploader("Choose a JPEG file", type=["jpg", "jpeg"])
            if uploaded_file is not None:
                # Use st.columns() to create two columns
                col1, col2 = st.columns(2)
                # Display the uploaded image in the first column
                col1.subheader("Uploaded Image")
                col1.image(uploaded_file, caption='Uploaded Image')

                # Perform prediction and display results in the second column
                col2.subheader("Predictions")
                predictions = img_pred(uploaded_file)
                col2.header(predictions)
        elif upload_option == "Sample Input":
            # Dropdown menu for selecting sample input
            sample_folders = os.listdir("sample_input_jpeg")
            selected_sample = st.sidebar.selectbox("Select Sample Input", sample_folders)
            
            if selected_sample:
                jpeg_path = load_sample_data_jpeg(selected_sample)
                # Process the sample input
                col1, col2 = st.columns(2)
                # Display the uploaded image in the first column
                col1.subheader("Sample Image")
                col1.image(jpeg_path, caption='Sample Image')
                col2.subheader("Predictions")
                predictions = img_pred(jpeg_path)
                col2.header(predictions)
            
    elif file_type == ".nii":
        st.header('NIfTI File Prediction and 3D Visualization')
        st.markdown("---")
        # Sidebar menu for NIfTI file uploads
        st.sidebar.header("Upload NIfTI Files or Choose Sample Input")
        upload_option = st.sidebar.radio("Select Option", [ "Sample Input","Upload Files"])

        if upload_option == "Upload Files":
            # File uploads for NIfTI
            nii_file_1 = st.sidebar.file_uploader("Upload NIfTI Image file 1 (Flair)", type=["nii"])
            nii_file_2 = st.sidebar.file_uploader("Upload NIfTI Segmentation file 2 (Seg)", type=["nii"])
            
            if nii_file_1 is not None and nii_file_2 is not None:
                st.sidebar.success(f"Files uploaded successfully")
                # Save the uploaded files to the UPLOAD_FOLDER
                nii_path_1 = os.path.join(UPLOAD_FOLDER, nii_file_1.name)
                nii_path_2 = os.path.join(UPLOAD_FOLDER, nii_file_2.name)

                with open(nii_path_1, "wb") as f:
                    f.write(nii_file_1.getvalue())

                with open(nii_path_2, "wb") as f:
                    f.write(nii_file_2.getvalue())

                # Process the uploaded files
                process_uploaded_files(nii_path_1, nii_path_2)

        elif upload_option == "Sample Input":
            # Dropdown menu for selecting sample input
            sample_folders = os.listdir("sample_input_nii")
            selected_sample = st.sidebar.selectbox("Select Sample Input", sample_folders)
            
            if selected_sample:
                flair_path, seg_path = load_sample_data(selected_sample)
                # Process the sample input
                process_uploaded_files(flair_path, seg_path)

    # Clear the contents of the UPLOAD_FOLDER
    clear_upload_folder()

def clear_upload_folder():
    # Clear the contents of the UPLOAD_FOLDER
    for file_name in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            # st.error(f"Error clearing the UPLOAD_FOLDER: {e}")
            pass

def process_uploaded_files(flair_path, seg_path):
    # Create a layout with 3 columns and 2 rows for NIfTI files
    col1, col2, col3 = st.columns(3)

    # Display prediction in the second column and first row
    col1.header("Prediction")
    predictions_nii = predict_nii(nii_path=flair_path)
    col1.header(predictions_nii)
    
    # Display slice from Flair.nii in the first column and first row
    col2.subheader("Flair.nii Slice")
    col2.plotly_chart(slice_flair(flair_path,77))

    # Display GIF in the third column and first row
    col3.subheader("GIF")
    gif_path = generate_gif(flair_path)
    col3.image(gif_path, use_column_width=True)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    # Display 3D mesh of the brain in the first column and second row
    col1.subheader("3D Mesh of Brain")
    col1.plotly_chart(mesh_3d(flair_path, seg_path))

    # Display 3D mesh with affected area in the second column and second row
    col2.subheader("3D Mesh with Affected Area")
    col2.plotly_chart(mesh_3d_affected_area(flair_path, seg_path))

    # Display 3D mesh of only tumor in the third column and second row
    col3.subheader("3D Mesh of Only Tumor")
    col3.plotly_chart(mesh_3d_tumor(seg_path))
    st.markdown("---")
    st.subheader("Segments with different effects")
    col1= st.columns(1)
    segments_differnt_effects(flair_path, seg_path)

if __name__ == "__main__":
    main()
