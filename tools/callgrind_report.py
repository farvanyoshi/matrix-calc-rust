#!/usr/bin/env python3
"""Render actual Callgrind instruction attribution, not CPU-time stack samples.

Shape requirements: none. Input must be a completed Callgrind event-cost file.
"""
import argparse,collections,csv,html,json,pathlib,re,subprocess

def parse(path):
    """Shape requirements: none; read function self-cost and real call-edge cost."""
    names={};costs=collections.Counter();edges=collections.Counter();events=[];summary=[];positions=1;current=None;callee=None;pending=False
    lines=path.read_text(errors='replace').splitlines()
    for line in lines:
        if line.startswith('events:'):events=line.split()[1:]
        elif line.startswith('positions:'):positions=len(line.split())-1
        elif line.startswith('summary:'):summary=[int(v) for v in line.split()[1:]]
        match=re.match(r'(c?fn)=\((\d+)\)(?: (.*))?$',line)
        if match:
            kind,key,name=match.groups();key=int(key)
            if name is not None:names[key]=name
            if kind=='fn':current=key;pending=False
            else:callee=key
        elif line.startswith('calls='):pending=True
        elif current is not None and line and line[0] in '0123456789+*-':
            tokens=line.split();raw=tokens[positions:]
            if not raw or not raw[0].isdigit():continue
            amount=int(raw[0])
            if pending:
                if callee is not None:edges[(current,callee)]+=amount
                pending=False
            else:costs[current]+=amount
    return names,costs,edges,events,summary

def main():
    """Shape requirements: none; output diagrams explicitly label instruction costs."""
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');args=p.parse_args()
    source=pathlib.Path(args.input);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    names,costs,edges,events,summary=parse(source)
    measured=summary[0] if summary else sum(costs.values());total=sum(costs.values())
    rows=[dict(function=names.get(key,f'function #{key}'),self_instructions=value,fraction=value/total if total else 0)for key,value in costs.most_common()]
    (out/'summary.json').write_text(json.dumps(dict(source=source.name,measurement='Callgrind instruction events; not wall-clock CPU samples',events=dict(zip(events,summary)),self_cost_sum=total,summary_instructions=measured,functions=rows),indent=2)+'\n')
    with(out/'functions.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=['function','self_instructions','fraction']);writer.writeheader();writer.writerows(rows)
    # A two-level flame-style chart is deliberately flat: no nonexistent stack
    # ancestry is inferred from aggregate callgraph edges.
    width=1400;left=20;usable=width-40;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="205" viewBox="0 0 {width} 205">', '<rect width="100%" height="100%" fill="white"/>','<style>text{font-family:monospace;font-size:12px} .title{font-size:17px;font-weight:bold}</style>', '<text x="20" y="27" class="title">Callgrind instruction-attribution flame chart (flat self-cost)</text>','<text x="20" y="52">Widths are executed instruction counts, not CPU time or inferred call stacks.</text>']
    chosen=rows[:60]
    rest=sum(row['self_instructions']for row in rows[60:])
    if rest:chosen=chosen+[dict(function='other measured functions',self_instructions=rest)]
    x=left
    for i,row in enumerate(chosen):
        w=usable*row['self_instructions']/total if total else 0;label=html.escape(row['function']);color=f'hsl({12+(i%9)*4},85%,{58+(i%4)*5}%)'
        parts.append(f'<g><title>{label}: {row["self_instructions"]:,} instructions</title><rect x="{x:.4f}" y="89" width="{w:.4f}" height="43" fill="{color}" stroke="white"/>')
        if w>50:parts.append(f'<text x="{x+4:.4f}" y="114">{html.escape(row["function"][:max(1,int(w/8)-1)])}</text>')
        parts.append('</g>');x+=w
    parts.append(f'<rect x="20" y="134" width="{usable}" height="35" fill="#e8b27a"/><text x="30" y="157">all collected function self-cost: {total:,} instructions</text><text x="20" y="191">See functions.csv / summary.json and callgraph.svg for named costs and measured edges.</text></svg>')
    (out/'instructions.svg').write_text('\n'.join(parts)+'\n')
    ids={key for key,_ in costs.most_common(24)};dot=['digraph costs {','rankdir=LR;','node [shape=box,fontname="monospace",fontsize=10];']
    for key in ids:dot.append(f'n{key} [label={json.dumps(names.get(key,str(key))+chr(10)+str(costs[key])+" self instructions")}];')
    for (a,b),cost in edges.most_common():
        if a in ids and b in ids:dot.append(f'n{a} -> n{b} [label="{cost} inclusive instructions"];')
    dot.append('}');(out/'callgraph.dot').write_text('\n'.join(dot)+'\n')
    try:subprocess.run(['dot','-Tsvg',str(out/'callgraph.dot'),'-o',str(out/'callgraph.svg')],check=True,capture_output=True)
    except FileNotFoundError:pass
    print(json.dumps(dict(profile=str(source),instructions=measured,self_cost_sum=total,top_functions=rows[:5],svg=str(out/'instructions.svg')),indent=2))
if __name__=='__main__':main()
