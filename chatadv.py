# chatadv_app.py
import streamlit as st
from lib.utils import load_environment, setup_rag_chain, create_pdf

# Load environment and setup RAG chain
load_environment()
rag_chain_with_source = setup_rag_chain()

# Streamlit app configuration
st.set_page_config(page_title="ChatAdv: The FSB Advising AI-Powered Chatbot", layout="wide", page_icon='🤖')

# Custom CSS
st.markdown("""
<style>
    .user-question {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .chatadv-answer {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🤖 ChatAdv: The FSB Advising AI-Powered Chatbot")

# Sidebar
st.sidebar.markdown("### About ChatAdv")
st.sidebar.info(
    "ChatAdv is an AI-powered chatbot designed to assist Farmer School of Business (FSB) "
    "students with their advising questions. It provides information based on FSB policies "
    "and procedures."
)

st.sidebar.markdown("### Maintained By")
st.sidebar.markdown("[Dr. Fadel Megahed](https://miamioh.edu/fsb/directory/?up=/directory/megahefm)")

st.sidebar.markdown("### Version")
st.sidebar.markdown("1.0.0 (Aug 01, 2024)")

# Toggle button for Disclaimers & References
if 'show_info' not in st.session_state:
    st.session_state.show_info = False

if st.sidebar.button('Toggle Disclaimers & References'):
    st.session_state.show_info = not st.session_state.show_info

if st.session_state.show_info:
    st.sidebar.markdown("""
    ### Disclaimers
    - ChatAdv is meant as a preparatory tool to synthesize information from multiple webpages prior to meeting with your academic advisor.
    - Always use ChatAdv at your own risk and evaluate the accuracy of the generated answers.
    - This tool does not replace personalized advising from your assigned academic advisor.
    
    ### References
    - [Farmer School of Business](https://miamioh.edu/fsb/)
    - [FSB Academic Advising](https://miamioh.edu/fsb/student-resources/academic-advising/)
    - [Miami University Academic Calendar](https://miamioh.edu/academic-calendar/)
    """)

# Help button
with st.sidebar.expander("Help & Instructions"):
    st.markdown("""
    1. Type your advising question in the text box.
    2. Click 'Ask ChatAdv' or press Enter to submit your question.
    3. Wait for ChatAdv to generate a response.
    4. Review the answer and any relevant quotes provided.
    5. Use the 'Export Chat to PDF' button to save your conversation.
    
    Remember: Always verify important information with your academic advisor.
    """)

# Main chat interface
st.markdown("### Ask your question:")
user_input = st.text_input("Type your question here:")

if user_input:
    with st.spinner('ChatAdv is thinking...'):
        chat_response = rag_chain_with_source.invoke(user_input)
        chat_answer = chat_response['answer'].split("<answer>")[1].split("</answer>")[0]

    # Display the question and answer with custom styling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='user-question'><strong>You:</strong> {user_input}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chatadv-answer'><strong>ChatAdv:</strong> {chat_answer}</div>", unsafe_allow_html=True)

    # Update chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": chat_answer})

# Display chat history
# if 'chat_history' in st.session_state and st.session_state.chat_history:
#     st.markdown("### Chat History:")
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             st.markdown(f"<div class='user-question'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
#         else:
#             st.markdown(f"<div class='chatadv-answer'><strong>ChatAdv:</strong> {message['content']}</div>", unsafe_allow_html=True)

# Export chat button
with st.expander("Export Your Entire Chat History to a PDF File"):
    user_name = st.text_input("Enter your name:")
    if user_name:
      user_name_cleaned = user_name.replace(" ", "_")
      # also remove punctuation and special characters
      user_name_cleaned = ''.join(e for e in user_name_cleaned if e.isalnum())
    
    if user_name:
        if st.button('Generate PDF'):
            with st.spinner('Generating PDF...'):
                pdf_output_path = create_pdf(st.session_state.chat_history, user_name)
            
            with open(pdf_output_path, "rb") as file:
                st.download_button(
                    label="Download PDF",
                    data=file,
                    file_name=f"{user_name_cleaned}_chatadv_advising_session.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
