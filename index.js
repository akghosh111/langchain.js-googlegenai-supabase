import 'dotenv/config'; 
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";

import { StringOutputParser } from "@langchain/core/output_parsers";


import { createClient } from "@supabase/supabase-js";




const apiKey = process.env.GOOGLE_API_KEY

const embeddings = new GoogleGenerativeAIEmbeddings({apiKey})
const sbApiKey = process.env.SUPABASE_API_KEY
const sbUrl = process.env.SUPABASE_URL
const client = createClient(sbUrl, sbApiKey)

const vectorStore = new SupabaseVectorStore(embeddings, {
    client,
    tableName: "match_documents",
    queryName: "match_documents"
})

const retriever = vectorStore.asRetriever()


const llm = new ChatGoogleGenerativeAI({ apiKey })

/**
 * Challenge:
 * 1. Create a prompt to turn a user's question into a 
 *    standalone question. (Hint: the AI understands 
 *    the concept of a standalone question. You don't 
 *    need to explain it, just ask for it.)
 * 2. Create a chain with the prompt and the model.
 * 3. Invoke the chain remembering to pass in a question.
 * 4. Log out the response.
 * **/

// A string holding the phrasing of the prompt
const standaloneQuestionTemplate = "Given a question, convert it into standalone question. question: {question} standalone question:";

// A prompt created using PromptTemplate and the fromTemplate method
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

// Take the standaloneQuestionPrompt and PIPE the model
const standaloneQuestionChain = standaloneQuestionPrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriever)

// Await the response when you INVOKE the chain. 
// Remember to pass in a question.
const response = await standaloneQuestionChain.invoke({
    question: "What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful."
})

console.log(response)

