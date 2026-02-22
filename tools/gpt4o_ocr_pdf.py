from collections.abc import Generator
import io
import base64
from typing import Any
import fitz  # PyMuPDF
from openai import OpenAI

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class Gpt4oOcrPdfTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        if not self.runtime or not self.runtime.credentials:
            raise Exception("Tool runtime or credentials are missing")
        
        # Get credentials (updated for any OpenAI-compatible VLM)
        api_key = str(self.runtime.credentials.get("api_key", "")).strip()
        api_base = str(self.runtime.credentials.get("api_base", "https://api.openai.com/v1")).strip()
        model = str(self.runtime.credentials.get("model", "qwen3-vl:32b")).strip()
        
        if not api_key:
            raise ValueError("API key is missing")
        if not api_base:
            raise ValueError("API Base URL is missing")
        
        # Get file
        file = tool_parameters.get("upload_file")
        if not file:
            raise ValueError("PDF file is required")
        
        file_binary = io.BytesIO(file.blob)
        
        # Extract images from PDF
        pdf_document = fitz.open(stream=file_binary, filetype="pdf")
        
        # System prompt for perfect invoice OCR
        system_message = """You are an expert OCR system specialized in invoices. 
Extract ALL text with perfect accuracy. Preserve exact structure:
- Invoice number, date, supplier name, VAT number
- All line items (description, quantity, unit price, total)
- Totals, VAT, grand total
Output ONLY clean Markdown. Use tables for line items. Never add explanations."""

        all_images = []
        all_markdown = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High res
            img_bytes = pix.tobytes(output="png")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            all_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
            
            if len(all_images) >= 5 or page_num == len(pdf_document) - 1:
                client = OpenAI(api_key=api_key, base_url=api_base)
                
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        {"type": "text", "text": "OCR these invoice pages and return perfect Markdown."},
                        *all_images
                    ]}
                ]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8000
                )
                
                markdown_content = response.choices[0].message.content or ""
                
                # Clean markdown code blocks
                if markdown_content.startswith("```"):
                    markdown_content = markdown_content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                
                all_markdown.append(markdown_content)
                all_images = []
        
        final_markdown = "\n\n---\n\n".join(all_markdown)
        yield self.create_text_message(final_markdown)
