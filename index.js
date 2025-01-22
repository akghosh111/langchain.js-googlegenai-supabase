import 'dotenv/config';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";


const apiKey = process.env.GOOGLE_API_KEY

const llm = new ChatGoogleGenerativeAI({apiKey})

const tweetTemplate = "Generate a promotional tweet for a product, from this product description:{productDesc}"

const tweetPrompt = PromptTemplate.fromTemplate(tweetTemplate)

const tweetChain = tweetPrompt.pipe(llm)

const response = await tweetChain.invoke({productDesc: "Electric shoes"})

console.log(response.content)