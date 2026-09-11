"""Render the editable Markdown research draft as a readable review PDF."""
from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,Preformatted
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
HERE=Path(__file__).resolve().parent
fontroot=Path('/usr/share/fonts/truetype/dejavu')
for face,file in [('Draft','DejaVuSerif.ttf'),('DraftBold','DejaVuSerif-Bold.ttf'),('DraftItalic','DejaVuSerif-Italic.ttf')]:
    pdfmetrics.registerFont(TTFont(face,str(fontroot/file)))
pdfmetrics.registerFontFamily('Draft',normal='Draft',bold='DraftBold',italic='DraftItalic',boldItalic='DraftBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='DraftBody',fontName='Draft',fontSize=9.5,leading=12.8,spaceAfter=7,alignment=TA_LEFT))
styles.add(ParagraphStyle(name='DraftTitle',fontName='DraftBold',fontSize=18,leading=22,spaceAfter=15))
styles.add(ParagraphStyle(name='DraftSection',fontName='DraftBold',fontSize=11.5,leading=14.5,spaceBefore=7,spaceAfter=9,keepWithNext=True))
styles.add(ParagraphStyle(name='DraftCaption',fontName='Draft',fontSize=8,leading=10.5,spaceAfter=9))
styles.add(ParagraphStyle(name='DraftTable',fontName='Draft',fontSize=8.2,leading=10.2))
styles.add(ParagraphStyle(name='DraftCode',fontName='Courier',fontSize=8.1,leading=11,spaceBefore=3,spaceAfter=9))


def inline(s):
    s=html.escape(s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'<link href="%s" color="#245579">%s</link>'%(m.group(2),m.group(1)),s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',s)
    return s


def render():
    lines=(HERE/'WORKSHOP_DRAFT.md').read_text().splitlines()
    story=[];i=0
    width=A4[0]-92
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- PAGEBREAK -->':story.append(PageBreak());i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(block),styles['DraftCode']));i+=1;continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['DraftTitle']));i+=1;continue
        if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['DraftSection']));i+=1;continue
        if line.startswith('!['):
            m=re.match(r'!\[(.*)\]\((.*)\)',line)
            path=HERE/m.group(2)
            w,h=PILImage.open(path).size
            iw=width
            story.append(Image(str(path),width=iw,height=iw*h/w))
            story.append(Paragraph(inline(m.group(1)),styles['DraftCaption']));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):
                    rows.append([Paragraph(inline(c),styles['DraftTable']) for c in cells])
                i+=1
            n=len(rows[0]);widths=[width*.55]+[width*.45/(n-1)]*(n-1)
            table=Table(rows,colWidths=widths,hAlign='LEFT',repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),
                ('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.gray),
                ('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,9)]);continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','![','<!--')):
            para.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(para)),styles['DraftBody']))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Draft',8)
        canvas.drawCentredString(A4[0]/2,24,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(HERE/'WORKSHOP_DRAFT.pdf'),pagesize=A4,
        leftMargin=46,rightMargin=46,topMargin=38,bottomMargin=38,
        title='Reference Sensitivity and Adaptive Tail Forecasting in Korean USDT Markets',author='',subject='Research draft')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('PDF rendered:',HERE/'WORKSHOP_DRAFT.pdf')


if __name__=='__main__':render()
