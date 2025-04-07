# Compile PhD Thesis 
#
# Run
# `make all`
# to build from source.
BUILD_DIR=$(PWD)
PDFLATEX=pdflatex -output-directory $(BUILD_DIR)
BIBLATEX=bibtex 

main.pdf: 
	$(PDFLATEX) main_jai.tex $(BUILD_DIR); $(BIBLATEX) main_jai; $(PDFLATEX) main_jai.tex; $(PDFLATEX) main_jai.tex;

clean:
	rubber --clean main_jai.tex
	rm main_jai.pdf
