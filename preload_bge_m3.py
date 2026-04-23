from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("Starting explicit download of BAAI/bge-m3...")
model = SentenceTransformer("BAAI/bge-m3")
print("Download complete and cached in HuggingFace cache!")
