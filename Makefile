.PHONEY:default

default: ms.pdf

ms.pdf: ms.tex
	pdflatex $<
	pdflatex $<
#	bibtex $<
#	pdflatex $<


.PHONEY:clean
clean:
	$(RM) main *.o *.txt *.ppl *.pdf *.png
