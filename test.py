from pdf2image import convert_from_path
pages = convert_from_path("C:\\Users\\vaibh\\Documents\\Xbiz_tasks\\Day2\\samples\\Doc1.pdf", dpi=300)
print("Pages:", len(pages))
