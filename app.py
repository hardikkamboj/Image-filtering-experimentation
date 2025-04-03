import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

def main():

    st.set_page_config(layout="wide")

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
        # st.subheader("Custom Kernel Values")
        
        # # Create a grid of number inputs for the kernel
        # kernel = np.copy(st.session_state.kernel)
        # cols = st.columns(kernel_size)
        
        # for i in range(kernel_size):
        #     for j in range(kernel_size):
        #         with cols[j]:
        #             kernel[i, j] = st.number_input(
        #                 f"[{i},{j}]",
        #                 value=float(kernel[i, j]),
        #                 format="%.2f",
        #                 key=f"k_{i}_{j}"
        #             )
        
        # # Update the kernel in session state
        # st.session_state.kernel = kernel
        
        # # Display the kernel as a matrix
        # st.subheader("Current Kernel")
        # st.text(np.array2string(kernel, precision=2))

        st.subheader("Custom Kernel Values")
        
        # Create two columns - one for kernel inputs, one for current kernel display
        kernel_col1, kernel_col2 = st.columns([3, 2])
        
        with kernel_col1:
            # Create a grid of number inputs for the kernel
            kernel = np.copy(st.session_state.kernel)
            
            # Use custom CSS to make inputs smaller
            st.markdown("""
            <style>
            .stNumberInput input {
                padding: 0.3rem;
            }
            div[data-baseweb="input"] {
                width: 80px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create input grid
            for i in range(kernel_size):
                cols = st.columns(kernel_size)
                for j in range(kernel_size):
                    with cols[j]:
                        kernel[i, j] = st.number_input(
                            f"",  # Remove label to save space
                            value=float(kernel[i, j]),
                            format="%.2f",
                            key=f"k_{i}_{j}",
                            step=0.5,
                            label_visibility="collapsed"  # Hide the label completely
                        )
        
        # Update the kernel in session state
        st.session_state.kernel = kernel
        
        # Display the kernel as a matrix
        with kernel_col2:
            st.markdown("**Current Kernel:**")
            st.text(np.array2string(kernel, precision=2))


        # Display original image
        st.subheader("Original Image")
        button_pressed = st.button("Apply Filter")

        outputcol_1, outputcol_2 = st.columns(2)

        with outputcol_1:
            st.image(image, use_container_width=True)


        
        # Apply filter button
        if button_pressed:
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
            with outputcol_2:
                #st.subheader("Filtered Image")
                st.image(filtered_img, use_container_width=True)
            
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