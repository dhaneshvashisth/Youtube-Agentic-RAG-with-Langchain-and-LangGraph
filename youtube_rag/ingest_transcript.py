from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_community.vectorstores import Qdrant


load_dotenv()

myclient = ChatOpenAI()


COLLECTION_NAME = "youtube_transcript"


def get_video_id(url: str):
    return url.split("v=")[1].split("&")[0]
    


def fetch_transcript(url):

    video_id = get_video_id(url)
    print(video_id)
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)

    

    text = " ".join([t["text"] for t in transcript])

    return text

def ingest_video(url):

    print("Loading transcript...")

    transcript = fetch_transcript(url)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript])

    print(f"Chunks created: {len(docs)}")

    embeddings = OpenAIEmbeddings()

    client = QdrantClient(":memory:")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )

    vectordb = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    vectordb.add_documents(docs)

    print("Embeddings stored in Qdrant")

    return vectordb


if __name__ == "__main__":

    url = input("Enter YouTube URL: ")

    ingest_video(url)