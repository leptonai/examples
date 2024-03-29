from leptonai.photon import Photon

from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import openai

import os
import gradio as gr


def create_retriever(target_file):
    loader = CSVLoader(target_file, csv_args={"delimiter": "\t"})
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256, chunk_overlap=0
    )
    docs = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    return db.as_retriever()


def create_qa_retrival_chain(target_file):
    foo_retriever = create_retriever(target_file)
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=foo_retriever
    )
    return qa


class EarningSage_Retriver(Photon):
    extra_files = {"AAPL-89728-report.tsv": "AAPL-89728-report.tsv"}

    requirement_dependency = ["tiktoken", "openai", "langchain", "chromadb", "gradio"]

    def init(self):
        os.environ["OPENAI_API_BASE"] = "API_BASE_FROM_TUNA"
        os.environ["OPENAI_API_KEY"] = "LEPTONAI_API_KEY"

        openai.api_base = os.environ["OPENAI_API_BASE"]
        openai.api_key = os.environ["OPENAI_API_KEY"]

        target_file = "AAPL-89728-report.tsv"

        print("Loading LLM from", openai.api_base)
        self.retrival_chain = create_qa_retrival_chain(target_file)
        print("Ready to serve!")

    @Photon.handler("chat")
    def chat(self, message):
        return self.retrival_chain.run(message)

    @Photon.handler(mount=True)
    def ui(self):
        blocks = gr.Blocks(title="🧙🏼 Earning Report Assistant")

        with blocks:
            gr.Markdown("# 🧙🏼 Earning Report Assistant")
            gr.Markdown("""
                This is an earning report assistant built for investors can't make the earning call on time. This sample is using Apple 2023 Q2 report. Feel free to reach out to uz@lepton.ai for more advanced features.
            """)
            with gr.Row():
                chatbot = gr.Chatbot(label="Model")
            with gr.Row():
                msg = gr.Textbox(
                    value=(
                        "What do you think of the relationship between Apple and it's"
                        " customers?"
                    ),
                    label="Questions you would like to ask",
                )

            with gr.Row():
                send = gr.Button("Send")
                clear = gr.Button("Clear")

            def respond_message(message, chat_history):
                bot_message = self.retrival_chain.run(message)
                chat_history.append((message, bot_message))
                return "", chat_history

            msg.submit(respond_message, [msg, chatbot], [msg, chatbot])
            send.click(respond_message, [msg, chatbot], [msg, chatbot])

            button1 = gr.Button(
                "Can you discuss the potential for further growth in the number of"
                " Apple devices per iPhone user?"
            )
            button2 = gr.Button("How is Apple ecosystem helping driving the revenue?")
            button3 = gr.Button("How is the feedback on Apple Pay Later?")

            def send_button_clicked(x):
                return gr.update(
                    value="""Can you discuss the potential for further growth in the number of Apple devices per iPhone user? Additionally, could you elaborate on how the monetization per user might vary between highly engaged "super users" and those who are not as deeply integrated into the Apple ecosystem?"""
                )

            def ask_ai_strategy(x):
                question = """What do you think of the relationship between Apple and it's customers? Could you give few examples on Apple trying to improve the customer relationship?"""
                return gr.update(value=question)

            def ask_pay_later(x):
                question = """Maybe as a quick follow-up, you talked about Apple Pay Later, how has the feedback been so far and how do you expect the adoption of our debt service over the next few quarters? Thank you."""
                return gr.update(value=question)

            button1.click(send_button_clicked, msg, msg)
            button2.click(ask_ai_strategy, msg, msg)
            button3.click(ask_pay_later, msg, msg)

            clear.click(lambda: None, None, chatbot, queue=False)

        return blocks
