#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.

from pylatex import Command
from pylatex.base_classes import Environment
import pylatex as ltx
from pylabyk import plt2


doc = plt2.LatexDoc(
    file_out='./demo_beamer/demo_beamer.pdf',
    documentclass=['beamer'],
)
doc.packages.append(ltx.Package('adjustbox'))

doc.append(Command('title', 'Sample title'))
doc.append(Command('frame', Command('titlepage')))

with doc.create(plt2.Frame()):
    doc.append(Command('frametitle', 'Table of Contents'))
    doc.append(Command('tableofcontents'))

with doc.create(plt2.Frame()):
    with doc.create(ltx.Table()):
        with doc.create(plt2.Adjustbox()):
            # doc.append(f'page {i}')
            plt2.latex_table(
                doc, [{'key': 'k', 'value': 'v'}] * 20)

with doc.create(plt2.Frame()):
    with doc.create(ltx.Table()):
        with doc.create(plt2.Adjustbox()):
            # doc.append(f'page {i}')
            plt2.latex_table(
                doc, [
                    {
                        f'k{i}': f'{i}'
                        for i in range(50)
                    }
                ])

doc.close()
print('--')

