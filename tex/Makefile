#LaTeX

.PHONEY:default

default:  main.pdf

main.pdf: main.tex
	pdflatex $<
	pdflatex $<
#	bibtex $<
#	pdflatex $<


.PHONEY:clean
clean:
	$(RM) main *.o *.txt *.ppl *.pdf *.png

