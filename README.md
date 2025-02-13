# COMCUDA Lab Sheets

These were converted to pandoc-markdown in 23-24.

They can be converted via pandoc, it uses a double pass as it includes templating.

```sh
# PDF
pandoc "name.md" --template "name.md" | pandoc -t latex -o "name.pdf"
# HTML
pandoc "name.md" --standalone --mathjax -o "name.html"
```

Either of these can be placed inside a batch file/sh file to automatically iterate files e.g.

```batch
@echo off
FOR %%f IN (*.md) DO (
    pandoc "%%~nf.md" --template "%%~nf.md" | pandoc -t latex -o "%%~nf.pdf"
)
```
