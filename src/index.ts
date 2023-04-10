import * as dotenv from "dotenv";

import { ConversationalRetrievalQAChain } from "langchain/chains";
import { OpenAIChat } from "langchain/llms";
import { GithubRepoLoader } from "langchain/document_loaders";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import readline from "readline";
import fs from "fs";

dotenv.config();

const githubstore = async (directory: string, github_url: string) => {
  const loader = new GithubRepoLoader(github_url, {
    branch: "main",
    recursive: false,
    unknown: "warn",
  });
  const docs = await loader.load();

  const splitter1 = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
  });
  const docOutput = await splitter1.splitDocuments(docs);
  const vectorStore = await HNSWLib.fromDocuments(
    docOutput,
    new OpenAIEmbeddings()
  );

  await vectorStore.save(directory);
};

const checkFilesExistInFolder = (
  folderPath: string,
  fileNames: string[]
): boolean => {
  const files = fs.readdirSync(folderPath);

  for (const fileName of fileNames) {
    if (!files.includes(fileName)) {
      return false;
    }
  }

  return true;
};

const run = async () => {
  const fileNames = ["args.json", "docstore.json", "hnswlib.index"];
  const githubUrl = process.argv[2];
  const directory = githubUrl
    .replace(/^https:\/\/github.com\//, "")
    .replace("/", "-");
  if (!fs.existsSync(directory)) {
    fs.mkdirSync(directory, { recursive: true });
  }
  if (!checkFilesExistInFolder(directory, fileNames)) {
    console.log("Not all files exist in the folder");
    console.log(`start to store ${githubUrl} information`);
    await githubstore(directory, githubUrl);
    console.log(`end store ${githubUrl} information`);
  }

  const model = new OpenAIChat({
    temperature: 0.9,
  });

  const loadedVectorStore = await HNSWLib.load(
    directory,
    new OpenAIEmbeddings()
  );

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    loadedVectorStore.asRetriever()
  );

  // To receive input from the user
  const read1 = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  let chatHistory = "";
  console.log("ask anything");
  read1.on("line", async (input) => {
    const question = input;

    const res = await chain.call({
      question,
      chat_history: chatHistory,
    });
    console.log(res);
    chatHistory += question + res.text;
    read1.prompt();
  });
};

run()
