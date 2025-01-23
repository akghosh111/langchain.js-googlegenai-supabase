import 'dotenv/config'; 
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";

import { StringOutputParser } from "@langchain/core/output_parsers";
import {
    RunnableParallel,
    RunnablePassthrough,
  } from "@langchain/core/runnables";

  document.addEventListener('submit', (e) => {
    e.preventDefault()
    progressConversation()
})


import { createClient } from "@supabase/supabase-js";




const apiKey = process.env.GOOGLE_API_KEY

const embeddings = new GoogleGenerativeAIEmbeddings({apiKey})
const sbApiKey = process.env.SUPABASE_API_KEY
const sbUrl = process.env.SUPABASE_URL
const client = createClient(sbUrl, sbApiKey)

const vectorStore = new SupabaseVectorStore(embeddings, {
    client,
    tableName: "documents",
    queryName: "match_documents"
})

function combineDocuments(docs){
    return docs.map((doc)=>doc.pageContent).join('\n\n')
}


const retriever = vectorStore.asRetriever()


const llm = new ChatGoogleGenerativeAI({ apiKey })




const standaloneQuestionTemplate = "Given a question, convert it into standalone question. question: {question} standalone question:";

const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer on the context provided. If you really don't know the answer, say 'I am sorry, I don't know the answer to that.' And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
question: {question}
answer:`

const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)


const standaloneQuestionChain = standaloneQuestionPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    
const retrieverChain = RunnableSequence.from([
    prevResult => prevResult.standalone_question,
    retriever,
    combineDocuments
])
const answerChain = answerPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())

const chain = RunnableSequence.from([
    {
        standalone_question: standaloneQuestionChain,
        original_input: new RunnablePassthrough()
    },
    {
        context: retrieverChain,
        question: ({ original_input }) => original_input.question
    },
    answerChain
])

async function progressConversation() {
    const userInput = document.getElementById('user-input')
    const chatbotConversation = document.getElementById('chatbot-conversation-container')
    const question = userInput.value
    userInput.value = ''


    const newHumanSpeechBubble = document.createElement('div')
    newHumanSpeechBubble.classList.add('speech', 'speech-human')
    chatbotConversation.appendChild(newHumanSpeechBubble)
    newHumanSpeechBubble.textContent = question
    chatbotConversation.scrollTop = chatbotConversation.scrollHeight
    const response = await chain.invoke({
        question: question
    })

    const newAiSpeechBubble = document.createElement('div')
    newAiSpeechBubble.classList.add('speech', 'speech-ai')
    chatbotConversation.appendChild(newAiSpeechBubble)
    newAiSpeechBubble.textContent = response
    chatbotConversation.scrollTop = chatbotConversation.scrollHeight
}


