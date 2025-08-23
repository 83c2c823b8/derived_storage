PDFDIR = pdf
PDFS   = $(wildcard $(PDFDIR)/*.pdf)
OUTPUT = index.html

$(OUTPUT): index.template.html $(PDFS)
	@echo "Generating $@ ..."
	@rm -f $@
	@while read -r line; do \
		if echo $$line | grep -q '<!-- PDF_LIST -->'; then \
			for f in $(PDFS); do \
				basename=$$(basename $$f); \
				echo "        <li><a href=\"$(PDFDIR)/$$basename\" target=\"_blank\">$$basename</a></li>"; \
			done; \
		else \
			echo "$$line"; \
		fi; \
	done < index.template.html > $@

.PHONY: clean
clean:
	rm -f $(OUTPUT)

