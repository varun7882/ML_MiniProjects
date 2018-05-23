import PyPDF2 as pd
f=open("SIFT.pdf","rb")
pdfread=pd.PdfFileReader(f)
pages=pdfread.numPages
print pages
wr=open("sift.txt","wb")
for i in range(pages):
    pg=pdfread.getPage(i)
    wr.write(pg.extractText())
wr.close()
    
