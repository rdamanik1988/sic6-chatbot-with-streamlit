# Overview
Repository ini dibuat sebagai materi LLM-Powered Chatbot with Streamlit. Pada repository ini digunakan streamlit sebagai user interface sehingga pengguna dapat berinteraksi langsung dengan RAG (Retrieval Augmented-Generation) yang telah dibuat. Repository ini menggunakan LangChain v0.3 dan llama3-8b-8192 sebagai modelnya.

# SetUp & Installation
Setup & Installation
1. Clone atau download repostiroy ini
```bash
git clone https://github.com/rdamanik1988/sic6-chatbot-with-streamlit.git
cd sic6-chatbot-with-streamlit
```
2. Install Dependencies:
```bash
pip install -r requirements.txt
//Make sure you have Python 3.7+ installed.
```
3. Set Environment Variables:
buat file dengan nama `.env` kemudian isi file tersebut dengan API Keys sebagai berikut
Example:
```bash
GROQ_API_KEY = "GROQQ API KEY"
```
atau ganti API KEY GROC di file utilities/rag.py pada sintaks
groq_api_key="<GANTI API KEY ANDA>"

4. Run Streamlit di local
```bash
streamlit run main.py
```
5. Lalu deploy ke [https://share.streamlit.io/](https://share.streamlit.io/)
