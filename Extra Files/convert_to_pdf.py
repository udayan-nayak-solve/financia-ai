#!/usr/bin/env python3
"""
Convert TECHNICAL_OVERVIEW.md to PDF with professional formatting
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

def convert_markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF with professional styling"""
    
    # Read the markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )
    
    # Create professional HTML document with CSS styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Financial AI Platform - Business Overview</title>
        <style>
            @page {{
                size: letter;
                margin: 1in;
                @bottom-center {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #666;
                }}
            }}
            
            body {{
                font-family: 'Georgia', 'Times New Roman', serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            
            h1 {{
                font-size: 24pt;
                font-weight: bold;
                color: #1a1a1a;
                margin-top: 24pt;
                margin-bottom: 12pt;
                page-break-after: avoid;
                border-bottom: 3px solid #0066cc;
                padding-bottom: 8pt;
            }}
            
            h2 {{
                font-size: 18pt;
                font-weight: bold;
                color: #0066cc;
                margin-top: 20pt;
                margin-bottom: 10pt;
                page-break-after: avoid;
            }}
            
            h3 {{
                font-size: 14pt;
                font-weight: bold;
                color: #004080;
                margin-top: 16pt;
                margin-bottom: 8pt;
                page-break-after: avoid;
            }}
            
            h4 {{
                font-size: 12pt;
                font-weight: bold;
                color: #333;
                margin-top: 12pt;
                margin-bottom: 6pt;
                page-break-after: avoid;
            }}
            
            p {{
                margin-bottom: 10pt;
                text-align: justify;
                orphans: 3;
                widows: 3;
            }}
            
            strong {{
                font-weight: bold;
                color: #1a1a1a;
            }}
            
            em {{
                font-style: italic;
            }}
            
            ul, ol {{
                margin-left: 20pt;
                margin-bottom: 10pt;
            }}
            
            li {{
                margin-bottom: 6pt;
            }}
            
            code {{
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                background-color: #f5f5f5;
                padding: 2pt 4pt;
                border-radius: 3pt;
            }}
            
            pre {{
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                background-color: #f5f5f5;
                padding: 10pt;
                border-left: 3pt solid #0066cc;
                margin: 10pt 0;
                overflow-x: auto;
                page-break-inside: avoid;
            }}
            
            hr {{
                border: none;
                border-top: 2px solid #ccc;
                margin: 20pt 0;
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10pt 0;
                page-break-inside: avoid;
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 8pt;
                text-align: left;
            }}
            
            th {{
                background-color: #0066cc;
                color: white;
                font-weight: bold;
            }}
            
            /* Avoid page breaks after headings */
            h1, h2, h3, h4, h5, h6 {{
                page-break-after: avoid;
            }}
            
            /* Keep sections together when possible */
            section {{
                page-break-inside: avoid;
            }}
            
            /* First page title styling */
            body > h1:first-of-type {{
                font-size: 28pt;
                text-align: center;
                border-bottom: none;
                margin-top: 100pt;
                margin-bottom: 40pt;
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
    print(f"  File size: {Path(pdf_file).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    md_file = "TECHNICAL_OVERVIEW.md"
    pdf_file = "TECHNICAL_OVERVIEW.pdf"
    
    try:
        convert_markdown_to_pdf(md_file, pdf_file)
    except Exception as e:
        print(f"✗ Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
