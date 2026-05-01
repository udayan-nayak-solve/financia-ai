#!/usr/bin/env python3
"""
Convert EXECUTIVE_SUMMARY.md to PDF - Condensed 2-3 page version
"""

import markdown
from weasyprint import HTML
from pathlib import Path

def convert_markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF with professional styling"""
    
    # Read the markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'toc']
    )
    
    # Create professional HTML document with CSS styling - condensed layout
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Financial AI Platform - Executive Summary</title>
        <style>
            @page {{
                size: letter;
                margin: 0.75in;
                @bottom-center {{
                    content: "Page " counter(page);
                    font-size: 9pt;
                    color: #666;
                }}
            }}
            
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                font-size: 10pt;
                line-height: 1.4;
                color: #333;
                max-width: 100%;
            }}
            
            h1 {{
                font-size: 20pt;
                font-weight: bold;
                color: #1a1a1a;
                margin-top: 16pt;
                margin-bottom: 8pt;
                page-break-after: avoid;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 6pt;
            }}
            
            h2 {{
                font-size: 14pt;
                font-weight: bold;
                color: #0066cc;
                margin-top: 14pt;
                margin-bottom: 6pt;
                page-break-after: avoid;
            }}
            
            h3 {{
                font-size: 11pt;
                font-weight: bold;
                color: #004080;
                margin-top: 10pt;
                margin-bottom: 4pt;
                page-break-after: avoid;
            }}
            
            h4 {{
                font-size: 10pt;
                font-weight: bold;
                color: #333;
                margin-top: 8pt;
                margin-bottom: 3pt;
                page-break-after: avoid;
            }}
            
            p {{
                margin-bottom: 6pt;
                text-align: justify;
                orphans: 2;
                widows: 2;
            }}
            
            strong {{
                font-weight: bold;
                color: #1a1a1a;
            }}
            
            em {{
                font-style: italic;
            }}
            
            ul, ol {{
                margin-left: 16pt;
                margin-bottom: 6pt;
            }}
            
            li {{
                margin-bottom: 3pt;
            }}
            
            hr {{
                border: none;
                border-top: 1px solid #ccc;
                margin: 12pt 0;
            }}
            
            /* Avoid page breaks after headings */
            h1, h2, h3, h4, h5, h6 {{
                page-break-after: avoid;
            }}
            
            /* First page title styling */
            body > h1:first-of-type {{
                font-size: 22pt;
                text-align: center;
                border-bottom: none;
                margin-top: 20pt;
                margin-bottom: 20pt;
                color: #0066cc;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    print(f"Converting {md_file} to PDF...")
    HTML(string=full_html).write_pdf(pdf_file)
    print(f"✓ PDF created successfully: {pdf_file}")
    
    # Get file size and page estimate
    file_size_kb = Path(pdf_file).stat().st_size / 1024
    print(f"  File size: {file_size_kb:.1f} KB")

if __name__ == "__main__":
    md_file = "EXECUTIVE_SUMMARY.md"
    pdf_file = "EXECUTIVE_SUMMARY.pdf"
    
    try:
        convert_markdown_to_pdf(md_file, pdf_file)
    except Exception as e:
        print(f"✗ Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
