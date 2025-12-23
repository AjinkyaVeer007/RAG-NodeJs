import dotenv from "dotenv";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";

dotenv.config();

const pdfPath = "./life.pdf";
const COLLECTION = "insurance";

async function main() {
  try {
    console.log("📄 Loading PDF...");
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load();

    console.log(`📑 Pages loaded: ${docs.length}`);

    console.log("✂️ Splitting into chunks...");
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 700,
      chunkOverlap: 100,
    });

    const chunks = await splitter.splitDocuments(docs);
    console.log(`🧩 Total chunks: ${chunks.length}`);

    console.log("🧠 Initializing embeddings...");
    const embeddings = new OpenAIEmbeddings({
      model: "text-embedding-3-small",
    });

    console.log("📦 Connecting to Qdrant...");
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: "http://localhost:6333",
        collectionName: COLLECTION,
      }
    );

    const BATCH_SIZE = 25;

    console.log("🚀 Starting ingestion...");
    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      const batch = chunks.slice(i, i + BATCH_SIZE);

      await vectorStore.addDocuments(batch);

      console.log(
        `✅ Embedded ${Math.min(i + BATCH_SIZE, chunks.length)} / ${
          chunks.length
        }`
      );
    }

    console.log("🎉 Indexing completed successfully!");
  } catch (err) {
    console.error("❌ Indexing failed:", err);
  }
}

main();
