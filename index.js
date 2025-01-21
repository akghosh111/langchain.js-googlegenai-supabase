import 'dotenv/config';
import fs from "fs/promises";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient } from '@supabase/supabase-js';
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';

try {
    // Read the local file
    const text = await fs.readFile("scrimba_info.txt", "utf-8");

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        separators: ["\n\n", "\n", " ", ""], 
        chunkOverlap: 50
    });

    const output = await splitter.createDocuments([text]);

    const sbApiKey = process.env.SUPABASE_API_KEY
    const sbUrl = process.env.SUPABASE_URL
    const apiKey = process.env.GOOGLE_API_KEY

    const client = createClient(sbUrl, sbApiKey)

    await SupabaseVectorStore.fromDocuments(
        output,
        new GoogleGenerativeAIEmbeddings({ apiKey }),
        {
            client,
            tableName: "documents",
        }
    )

    console.log(output);
} catch (err) {
    console.error(err);
}
