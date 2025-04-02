import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

def main():
    st.title("Image Filtering Application")
    st.write("Upload an image and apply custom convolution kernels")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert to grayscale if image is RGB
        if len(img_array.shape) > 2 and img_array.shape[2] == 3:
            has_color = True
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            has_color = False
            img_gray = img_array
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Kernel size selection
        kernel_size = st.selectbox("Select Kernel Size", [3, 5, 7], index=0)
        
        # Initialize kernel with zeros
        if 'kernel' not in st.session_state or st.session_state.kernel_size != kernel_size:
            # Initialize with identity kernel (1 in center, 0 elsewhere)
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            kernel[center, center] = 1
            st.session_state.kernel = kernel
            st.session_state.kernel_size = kernel_size
        
        # Pre-defined kernels
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("Identity"):
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                kernel[center, center] = 1
                st.session_state.kernel = kernel
        
        with col2:
            if st.button("Blur"):
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
                st.session_state.kernel = kernel
        
        with col3:
            if st.button("Sharpen"):
                if kernel_size == 3:
                    kernel = np.array([
                        [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]
                    ])
                else:
                    kernel = np.ones((kernel_size, kernel_size)) * -1
                    center = kernel_size // 2
                    kernel[center, center] = kernel_size * kernel_size
                st.session_state.kernel = kernel
        
        with col4:
            if st.button("Edge"):
                if kernel_size == 3:
                    kernel = np.array([
                        [-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]
                    ])
                else:
                    kernel = np.ones((kernel_size, kernel_size)) * -1
                    center = kernel_size // 2
                    kernel[center, center] = kernel_size * kernel_size - 1
                st.session_state.kernel = kernel
        
        with col5:
            if st.button("Emboss"):
                if kernel_size == 3:
                    kernel = np.array([
                        [-2, -1, 0],
                        [-1, 1, 1],
                        [0, 1, 2]
                    ])
                else:
                    kernel = np.zeros((kernel_size, kernel_size))
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            kernel[i, j] = j - i
                st.session_state.kernel = kernel
        
        # Custom kernel input
        st.subheader("Custom Kernel Values")
        
        # Create a grid of number inputs for the kernel
        kernel = np.copy(st.session_state.kernel)
        cols = st.columns(kernel_size)
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                with cols[j]:
                    kernel[i, j] = st.number_input(
                        f"[{i},{j}]",
                        value=float(kernel[i, j]),
                        format="%.2f",
                        key=f"k_{i}_{j}"
                    )
        
        # Update the kernel in session state
        st.session_state.kernel = kernel
        
        # Display the kernel as a matrix
        st.subheader("Current Kernel")
        st.text(np.array2string(kernel, precision=2))
        
        # Apply filter button
        if st.button("Apply Filter"):
            # Apply the convolution
            if has_color:
                # Process each channel separately
                r = cv2.filter2D(img_array[:,:,0], -1, kernel)
                g = cv2.filter2D(img_array[:,:,1], -1, kernel)
                b = cv2.filter2D(img_array[:,:,2], -1, kernel)
                
                # Combine channels
                filtered_img = np.stack([r, g, b], axis=2)
                
                # Clip values to valid range
                filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
            else:
                # Apply directly to grayscale
                filtered_img = cv2.filter2D(img_gray, -1, kernel)
                filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
            
            # Display the filtered image
            st.subheader("Filtered Image")
            st.image(filtered_img, use_column_width=True)
            
            # Add download button for filtered image
            filtered_pil = Image.fromarray(filtered_img)
            buf = BytesIO()
            filtered_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download filtered image",
                data=byte_im,
                file_name="filtered_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()