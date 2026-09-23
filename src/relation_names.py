"""Readable relation labels for offline reflection prompts."""
import re


WIKI_LABELS = {
    'P1376': 'capital of', 'P131': 'located in the administrative territorial entity',
    'P2962': 'title of chess person', 'P579': 'IMA status and/or rank',
    'P512': 'academic degree', 'P190': 'twinned administrative body',
    'P17': 'country', 'P6': 'head of government', 'P26': 'spouse',
    'P463': 'member of', 'P108': 'employer', 'P102': 'member of political party',
    'P27': 'country of citizenship', 'P1435': 'heritage designation',
    'P39': 'position held', 'P31': 'instance of', 'P1346': 'winner',
    'P166': 'award received', 'P150': 'contains the administrative territorial entity',
    'P551': 'residence', 'P1411': 'nominated for', 'P793': 'significant event',
    'P54': 'member of sports team', 'P69': 'educated at',
}


def prompt_mapping(dataset, source, output):
    if dataset not in {'WIKI', 'YAGO'}:
        return source
    rows = [line.rsplit(maxsplit=1) for line in source.read_text().splitlines() if line.strip()]
    names = []
    for original, rid in sorted(rows, key=lambda r: int(r[1])):
        if dataset == 'WIKI':
            label = f'{original}: {WIKI_LABELS[original]}'
        else:
            match = re.fullmatch(r'<([A-Za-z][A-Za-z0-9]*)>', original)
            if match is None:
                raise ValueError(f'Invalid YAGO relation: {original}')
            label = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', match[1]).lower() + f' [{original}]'
        names.append(f'{label}\t{rid}\n')
    output.write_text(''.join(names))
    return output
