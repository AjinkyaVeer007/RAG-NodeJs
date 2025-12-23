import dotenv from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import readline from "readline/promises";
import { stdin as input, stdout as output } from "process";
import { ChatOpenAI } from "@langchain/openai";

dotenv.config();

const COLLECTION = "insurance";
const rl = readline.createInterface({ input, output });

const client = new ChatOpenAI({
  model: "gpt-5",
  streaming: true,
});

async function main() {
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      collectionName: COLLECTION,
    }
  );

  while (true) {
    const userQuery = await rl.question("> ");

    const result = await vectorStore.similaritySearch(userQuery);

    let content = "";

    for (let i = 0; i < result.length; i++) {
      content =
        content +
        `\n Page no - ${result[i].metadata.loc.pageNumber}\n content - ${result[i].pageContent}\n`;
    }

    let systemPrompt = `
    You are an agent use for rag system. You will summerise the provide content and suggest page number for reference. You will never answer outside of content. If any user ask a question out of scope then politly replay like I am only answer your query that is releated to documents given to me.

    Content - ${content}

    if there is not content then just answer as cannot find relevent explaination of your query in provided document.
   `;

    console.log("Searching for query... Please wait...");
    const response = await client.stream([
      {
        role: "system",
        content: systemPrompt,
      },
      {
        role: "user",
        content: userQuery,
      },
    ]);

    process.stdout.write("");

    for await (const chunk of response) {
      const token = chunk.content;
      if (token) {
        process.stdout.write(token);
      }
    }

    process.stdout.write("\n\n");
  }
}

main();
