import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


# Load the TFLite models
model_paths = {
    'mango': 'mango_model.tflite',
    'soybean': 'soybean_model.tflite',
    'potato': 'potato_model.tflite'
}
interpreters = {}
for crop, path in model_paths.items():
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    interpreters[crop] = interpreter

# Define class names for each model
class_names = {
    'mango': ['Anthracnose', 'Dag Disease', 'Galls', 'Red Rust'],
    'soybean': ["bacterial_blight", "cercospora_leaf_blight", "downey_mildew", "frogeye", "potassium_deficiency", "soybean_rust"],
    'potato': ['Early Blight','Healthy', 'Late Blight', 'leaf-diseases']
}

class_descriptions = {
    'mango': {
        'Anthracnose': {
            'full_name': 'Anthracnose / আমের এনথ্রাকনোজ',
            'Symptoms': '১। আমের এনথ্রাকনোজ হলে কচি পাতায় অনিয়মিত দাগ দেখা যায় । ২। আমের মুকুল কালো হয়ে যায়, আমের গুটি ঝড়ে যায় ।',
            'Management': '১। সময়মত প্রুনিং করে গাছ ও বাগান পরিস্কার পরিচ্ছন্ন রাখা । ২। গাছের নিচে ঝড়ে পড়া পাতা, মুকুল বা আমের গুটি অপসারণ করা। ৩। কার্বেন্ডাজিম বা ম্যানকোজেব গ্রুপের যে কোন ছত্রাকনাশক যেমন: কমপ্যানিয়ন ২ গ্রাম বা এমকোজিন বা গোল্ডাজিম ১ গ্রাম / লি. হারে পানিতে মিশিয়ে ১৫ দিন পরপর ৩-৪ বার স্প্রে করা ।'
        },
        'Dag Disease': {
            'full_name': 'Dag Disease / আমের পাতার দাগ রোগ',
            'Symptoms': '১। আমের পাতায় কাল কোণাকৃতির দাগ দেখা যায় । ২। কচি কান্ড ও ফলেও দাগ দেখা যায় ।',
            'Management': '১। বাগান পরিচর্যার সময় গাছ ক্ষতিগ্রস্থ না করা । ২। জীবানুমুক্ত বীজ বা কলম রোপন করা। ৩। কপার অক্সিক্লোরাইড ২.৫ গ্রাম / লি. হারে পানিতে মিশিয়ে ১৫ দিন পরপর ৩-৪ বার স্প্রে করা ।'
        },
        'Galls': {
            'full_name': 'Gall Disease / আমের গল রোগ',
            'Symptoms': '১। এ পোকা কচি পাতায় আক্রমন করার ফলে পাতায় বিভিন্ন ধরনের গল তৈরী হয়। ২। অনেক সময় পাতা শুকিয়ে মারা যায় ।',
            'Management': '১। আক্রান্ত পাতা অপসারণ করা । ২। নতুন পাতা বের হবার পর ফেনিট্রিথিয়ন ২ মি.লি. / লি. হারে পানিতে মিশিয়ে স্প্রে করা।'
        },
        'Red Rust': {
            'full_name': 'Red Rust / আমের লাল মরিচা রোগ',
            'Symptoms': '১। এ রোগের আক্রমণে পাতায়, ফলে ও কান্ডে লালচে মরিচার মত একধরনের উচু দাগ দেখা যায়। ২। একধরণের সবুজ শৈবালের আক্রমণে এ রোগ হয়।',
            'Management': '১। আক্রান্ত পাতা ও ডগা ছাটাই করে ধ্বংস করা। ২। কপার বা কুপ্রাভিট ১০ লি. পানিতে ২০ গ্রাম মিশিয়ে ১৫ দিন পরপর ২ বার স্প্রে করা।'
        },
    },
    'soybean': {
        'bacterial_blight': {
            'full_name': 'Bacterial Blight / সয়াবিন ব্যাকটেরিয়াল ব্লাইট',
            'Symptoms': '১। পাতা, কান্ড, এবং শুঁটি কালো দাগের সাথে শুকিয়ে যায়। ২। গাছের বৃদ্ধি বন্ধ হয়ে যায় এবং ফলন কমে যায়।',
            'Management': '১। রোগমুক্ত বীজ ব্যবহার করা। ২। ফসলের আগাছা পরিস্কার করা। ৩। প্রয়োজনীয় ব্যবস্থাপনার জন্য অনুমোদিত ব্যাকটেরিসাইড ব্যবহার করা।'
        },
        'cercospora_leaf_blight': {
            'full_name': 'Cercospora Leaf Blight / সয়াবিন সারকোস্পোরা পাতা ঝলসানো',
            'Symptoms': '১। পাতায় ছোট বাদামী বা লালচে গোলাকার দাগ দেখা যায়। ২। গাছের বৃদ্ধি হ্রাস পায় এবং ফলন কম হয়।',
            'Management': '১। প্রয়োজনীয় ছত্রাকনাশক ব্যবহার করা। ২। রোগ প্রতিরোধী জাত ব্যবহার করা।'
        },
        'downey_mildew': {
            'full_name': 'Downy Mildew / সয়াবিন ডাউনি মিলডিউ',
            'Symptoms': '১। পাতার নিচের দিকে সাদা ছত্রাকের স্তর দেখা যায়। ২। পাতা হলুদ বা বাদামী হয়ে যায় এবং পড়ে যায়।',
            'Management': '১। রোগমুক্ত বীজ ব্যবহার করা। ২। প্রয়োজনীয় ফাঙ্গিসাইড ব্যবহার করা।'
        },
        'frogeye': {
            'full_name': 'Frogeye Leaf Spot / সয়াবিন ফ্রগআই পাতা দাগ',
            'Symptoms': '১। পাতার উপর গোলাকার ধূসর দাগ দেখা যায়। ২। দাগগুলি একত্রিত হয়ে বড় দাগ তৈরি করে এবং পাতা ঝলসানো হতে পারে।',
            'Management': '১। প্রতিরোধী জাতের সয়াবিন ব্যবহার করা। ২। অনুমোদিত ফাঙ্গিসাইড ব্যবহার করা।'
        },
        'potassium_deficiency': {
            'full_name': 'Potassium Deficiency / সয়াবিন পটাশিয়ামের ঘাটতি',
            'Symptoms': '১। পুরনো পাতার কিনারায় হলুদাভ দাগ দেখা যায়। ২। গাছের বৃদ্ধি কম হয় এবং ফলন হ্রাস পায়।',
            'Management': '১। পটাশ সার ব্যবহার করা। ২। মাটি পরীক্ষা করে সারের মাত্রা নির্ধারণ করা।'
        },
        'soybean_rust': {
            'full_name': 'Soybean Rust / সয়াবিন মরিচা রোগ',
            'Symptoms': '১। পাতার নিচের দিকে ছোট বাদামী বা লালচে দাগ দেখা যায়। ২। গাছ দুর্বল হয়ে যায় এবং ফলন কমে যায়।',
            'Management': '১। অনুমোদিত ফাঙ্গিসাইড স্প্রে করা। ২। সঠিক সময়ে ফসল কাটা।'
        },
    },
    'potato': {
        'Early Blight': {
            'full_name': 'Early Blight / আলুর আর্লি ব্লাইট',
            'Symptoms': '১। পাতায় ছোট বাদামী দাগ দেখা যায় যা আস্তে আস্তে বড় হয়। ২। গাছের কান্ডও ক্ষতিগ্রস্থ হয় এবং ফলন কমে যায়।',
            'Management': '১। আক্রান্ত পাতা ও ডগা ছাটাই করে ধ্বংস করা। ২। কপার বা কুপ্রাভিট ১০ লি. পানিতে ২০ গ্রাম মিশিয়ে ১৫ দিন পরপর ২ বার স্প্রে করা।'
        },
        'Healthy': {
            'full_name': 'Healthy / সুস্থ আলু গাছ',
            'Symptoms': '১। সুস্থ গাছে কোনো দাগ বা রোগের লক্ষণ থাকে না। ২। গাছ সবুজ ও সতেজ থাকে।',
            'Management': '১। রোগ প্রতিরোধী জাতের আলু গাছ ব্যবহার করা। ২। সঠিক সার প্রয়োগ ও পরিচর্যা করা।'
        },
        'Late Blight': {
            'full_name': 'Late Blight / আলুর লেট ব্লাইট',
            'Symptoms': '১। পাতায় বড় কালো বা বাদামী দাগ দেখা যায়। ২। গাছের কান্ডে ও কন্দে আক্রান্ত হয় এবং দ্রুত পচে যায়।',
            'Management': 'রিমিত সার ব্যবহার করা , সকালে সেচ প্রয়োগ করা ও উত্তম নিষ্কাষণ ব্যবস্থা রাখা। * সারির দুই পাশে বেশি করে মাটি তুলে দেয়া। •রোগের অনুকুল পূর্বাভাস (কুয়াশাচ্ছন্ন ও মেঘলা আবহাওয়ায় ) পাওয়া মাত্র প্রতিরোধক হিসাবে (ম্যানকোজেব + মেটালেক্সিল) জাতীয় ছত্রাকনাশক যেমন: রিডোমিল গোল্ড বা করমি বা মেটারিল বা ম্যানকোসিল বা ক্রেজি প্রতি লিটার পানিতে ২ গ্রাম হারে মিশিয়ে স্প্রে করা *রোগের আক্রমণ বেশী হলে প্রতি লিটার পানিতে-২ গ্রাম সিকিউর মিশিয়ে স্প্রে করলে ভাল ফল পাওয়া যায় ।'
        },
        'leaf-diseases': {
            'full_name': 'Leaf Diseases / আলুর পাতা সম্পর্কিত রোগ',
            'Symptoms': '১। পাতায় বিভিন্ন ধরনের দাগ দেখা যায় যা গাছের বৃদ্ধি ও ফলনে প্রভাব ফেলে। ২। রোগের ধরন অনুযায়ী দাগের রঙ ও আকার ভিন্ন হয়।',
            'Management': '১। নিয়মিত গাছ পর্যবেক্ষণ করা। ২। আক্রান্ত পাতা অপসারণ করা। ৩। অনুমোদিত ছত্রাকনাশক ব্যবহার করা।'
        },
    }
}


# Set up page configuration
st.set_page_config(page_title="Disease Detection", layout="centered")



# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

def navigate(page_name):
    st.session_state.current_page = page_name

st.markdown("""
    <style>
    .title { font-size: 30px; text-align: center; font-weight: bold; margin-bottom: 20px; color: #2E7D32; }
    .sub-title { font-size: 18px; text-align: center; margin-bottom: 20px; color: #388E3C; }
    </style>
""", unsafe_allow_html=True)

# Function to run detection for all crops
def run_detection(crop, image):
    interpreter = interpreters[crop]
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize image according to crop model
    if crop == 'mango':
        image = image.resize((640, 640))
    elif crop == 'soybean':
        image = image.resize((416, 416))
    elif crop == 'potato':
        image = image.resize((416, 416))  # Assuming same size for potato

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = np.squeeze(output_data, axis=0)

    bounding_boxes = output_data[:, :4]
    objectness_scores = output_data[:, 4]
    class_probabilities = output_data[:, 5:]

    threshold = 0.5
    valid_indices = np.where(objectness_scores > threshold)[0]

    if len(valid_indices) > 0:
        valid_bounding_boxes = bounding_boxes[valid_indices]
        valid_scores = objectness_scores[valid_indices]
        valid_class_probabilities = class_probabilities[valid_indices]

        predicted_classes = np.argmax(valid_class_probabilities, axis=1)
        predicted_class_name = class_names[crop][predicted_classes[0]]  # Take the first detected class

        # UI elements for displaying detection results
        st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold;'>{crop.capitalize()}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=f"ছবি: {predicted_class_name}", use_column_width=True)
        with col2:
            st.markdown(f"""
                <div style='border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f9f9f9;'>
                    <div style='font-size: 18px; color: #555; margin-bottom: 5px;'>Disease Name / (রোগের নাম): {class_descriptions[crop][predicted_class_name]['full_name']}</div>
                    <div style='font-size: 15px; color: #555; margin-bottom: 5px;'>Symptoms / (লক্ষণ): {class_descriptions[crop][predicted_class_name]['Symptoms']}</div>
                    <div style='font-size: 14px; color: #555; margin-bottom: 5px;'>Management / (ব্যাবস্থাপনা): {class_descriptions[crop][predicted_class_name]['Management']}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No objects detected with a high confidence score.")

def home_page():
    st.write("<div class='title'>Hawkeye ~ Eagle</div>", unsafe_allow_html=True)
    st.write("<div class='sub-title'>Click to Detect (রোগ জানাতে ট্যাপ করুন)</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Mango (আম)"):
            st.session_state["crop"] = "mango"
            navigate("camera")
        if st.button("Rice (ধান)"):
            st.session_state["crop"] = "ধান"
            navigate("camera")
        if st.button("Papaya (পেঁপে)"):
            st.session_state["crop"] = "পেঁপে"
            navigate("camera")
    with col2:
        if st.button("Lemon (লেবু)"):
            st.session_state["crop"] = "লেবু"
            navigate("camera")
        if st.button("Soybean (সয়াবিন)"):
            st.session_state["crop"] = "soybean"
            navigate("camera")
        if st.button("Potato (আলু)"):
            st.session_state["crop"] = "potato"
            navigate("camera")


    # Streamlit button to handle navigation
    if st.button("Next (পরবর্তী ধাপে যান)", key="next"):
        navigate("camera")

def camera_page():
    st.write(f"<div class='title'>{st.session_state.get('crop', 'Camera')}</div>", unsafe_allow_html=True)
    st.write("<div class='sub-title'>Capture or Upload Image</div>", unsafe_allow_html=True)
    st.write("<div class='sub-title'>(ছবি তুলুন বা একটি ছবি আপলোড করুন)</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style='text-align: center; font-weight: bold; font-size: 20px;'>Capture Image (ছবি তুলুন)</div>
            <div style='text-align: center;'>Click the button below to capture an image with the camera.</div>
            <div style='text-align: center;'>ক্যামেরা চালু করতে "Open Camera" বাটনে ক্লিক করুন</div>
        """, unsafe_allow_html=True)
        if st.button("Open Camera", key="capture"):
            captured_image = Image.new('RGB', (300, 300), color='gray')
            st.session_state["captured_image"] = captured_image
            st.image(captured_image, caption="Open Image")
            st.write("Image captured!")  
        if st.button("Back (পিছনে ফিরে যান)"):
            navigate("home")

    with col2:
        st.markdown("""
            <div style='text-align: center; font-weight: bold; font-size: 20px;'>Upload Image (ছবি আপলোড করুন)</div>
            <div style='text-align: center;'>Upload an image file from your device to display below.</div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image")
            st.session_state["uploaded_image"] = uploaded_image
        if st.button("Detect (সনাক্ত করুন)"):
            if "captured_image" in st.session_state or "uploaded_image" in st.session_state:
                navigate("detection")
            else:
                st.write("No image captured or uploaded to process!")

def detection_page():
    # st.title('ফলাফল')

    crop = st.session_state.get("crop")
    if crop and crop in interpreters:
        if "uploaded_image" in st.session_state:
            run_detection(crop, st.session_state["uploaded_image"])
        elif "captured_image" in st.session_state:
            run_detection(crop, st.session_state["captured_image"])
        else:
            st.write("No image available for detection.")
    else:
        st.write("No model available for the selected crop.")

        # Add Back and Next buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back (পিছনে ফিরে যান)"):
            navigate("camera")
    with col2:
        if st.button("Home (হোমে ফিরে যান)"):
            navigate("home")

if st.session_state.current_page == "home":
    home_page()
elif st.session_state.current_page == "camera":
    camera_page()
elif st.session_state.current_page == "detection":
    detection_page()
