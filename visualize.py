import nibabel as nib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.draw import ellipsoid
import matplotlib.pyplot as plt
from io import BytesIO
import gif
import nilearn as nl
import nilearn.plotting as nlplt


# 3d mesh of brain using flair.nii
def mesh_3d(nii_path,seg_path):
    img = nib.load(nii_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    
    verts, faces, normals, values = measure.marching_cubes(img, 1)
    x, y, z = verts.T
    i, j, k = faces.T

    mesh1 = go.Mesh3d(x=x, y=y, z=z,color='gray', opacity=0.5, i=i, j=j, k=k)
# make gr4ay dark 
    verts, faces, normals, values = measure.marching_cubes(seg, 2)
    x, y, z = verts.T
    i, j, k = faces.T

    mesh2 = go.Mesh3d(x=x, y=y, z=z, color='yellow', opacity=0.5, i=i, j=j, k=k)
    bfig = go.Figure(data=[mesh1,mesh2])
    bfig.update_layout(
                        autosize=False,
                        width=500,
                        height=500
                        
    )
    # bfig.show()
    return bfig

# mesh3d with affected area
def mesh_3d_affected_area(nii_path,seg_path)  :
    img = nib.load(nii_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    brain_parts = [
    {'img':img, 'color':'gray', 'level':0},
    {'img':seg, 'color':'purple', 'level':0},
    {'img':seg, 'color':'red', 'level':1},
    {'img':seg, 'color':'yellow', 'level':2},
    {'img':seg, 'color':'blue', 'level':3}
    ]

    meshes = []
    for part in brain_parts:
        print(part['color'],part['level'])
        verts, faces, normals, values = measure.marching_cubes(part['img'], part['level'])
        x, y, z = verts.T
        i, j, k = faces.T

        mesh = go.Mesh3d(x=x, y=y, z=z,color=part['color'], opacity=0.5, i=i, j=j, k=k)
        meshes.append(mesh)


    bfig = go.Figure(data=meshes)
    bfig.update_layout(
                        autosize=False,
                        width=500,
                        height=500
                        
    )
    # bfig.show()
    return bfig


# only tumor  
def mesh_3d_tumor(seg_path):
    seg = nib.load(seg_path).get_fdata()
    verts, faces, normals, values = measure.marching_cubes(seg, 2)
    x, y, z = verts.T
    i, j, k = faces.T

    mesh2 = go.Mesh3d(x=x, y=y, z=z, color='yellow', opacity=0.5, i=i, j=j, k=k)
    bfig = go.Figure(data=[mesh2])
    bfig.update_layout(
                        autosize=False,
                        width=500,
                        height=500
                        
    )
    return bfig


# Show segments of tumor using different effects
def segments_differnt_effects(nii_path,seg_path):
    niimg = nl.image.load_img(nii_path)
    nimask = nl.image.load_img(seg_path)

    fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


    nlplt.plot_anat(niimg,
                    title='plot_anat',
                    axes=axes[0])

    nlplt.plot_epi(niimg,
                title='plot_epi',
                axes=axes[1])

    nlplt.plot_img(niimg,
                title='plot_img',
                axes=axes[2])

    nlplt.plot_roi(nimask, 
                title='mask plot_roi',
                bg_img=niimg, 
                axes=axes[3], cmap='Paired')
    
    st.pyplot(fig)
    plt.close(fig)
 
# creating a gif from the flair.nii file
@gif.frame
def plot_slice(slice_data):
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')

def generate_gif(flair_nii_path):
    flair_img = nib.load(flair_nii_path)
    flair_data = flair_img.get_fdata()

    num_slices = flair_data.shape[-1]
    frames = []
    
    for i in range(num_slices):
        slice_data = flair_data[:, :, i]
        frame = plot_slice(slice_data)
        frames.append(frame)

    gif_path = flair_nii_path.replace('.nii', '_animation.gif')

    gif.save(frames, gif_path, duration=100)  # Adjust duration as needed

    return gif_path

def slice_flair(nii_path, slice_number):
    img = nib.load(nii_path).get_fdata()

    # Display the selected slice
    fig = px.imshow(img[:, :, slice_number], color_continuous_scale='gray')
    fig.update_layout(
        autosize=False,
        width=500,
        height=500
    )

    return fig