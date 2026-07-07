# Course Report PDF

This folder is a self-contained LaTeX project for the course report that borrows
the AAAI 2027 visual style. It is not intended for AAAI submission.

- `course_report_aaai2027.tex`: main report source
- `aaai2027-course.sty`: derived course-report style with XeLaTeX/CJK support
- `figures/`: report-local figures and cropped comparison images
- `dist/`: final PDF output
- `build/latex/`: LaTeX intermediate files

Build from the repository root:

```bash
docs/course_report_pdf/build.sh
```

The generated PDF is written to:

```text
docs/course_report_pdf/dist/course_report_aaai2027.pdf
```
