#!/usr/bin/env python3
"""Sample actual main-thread stacks with GDB/MI. Shape requirements: none.

Output is folded stacks and a raw MI transcript. This is an interrupting sampler,
not a hardware-counter profiler; each recorded stack counts as one sample.
"""
import argparse, collections, json, os, pathlib, re, select, subprocess, time

def main():
    """Shape requirements: executable arguments must select a finite workload."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--gdb',default='gdb')
    parser.add_argument('--output',required=True)
    parser.add_argument('--interval',type=float,default=0.02)
    parser.add_argument('--timeout',type=float,default=90)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.interval<=0:parser.error('positive interval and executable required')
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    child=subprocess.Popen([args.gdb,'--nx','--quiet','--interpreter=mi2','--args',*command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
    transcript=[];buffer=b'';counter=0;costs=collections.Counter();exited=False
    def send(text):
        nonlocal counter
        counter+=1;child.stdin.write(f'{counter}{text}\n'.encode());child.stdin.flush();return str(counter)
    def read_until(accept,limit):
        nonlocal buffer,exited
        while time.monotonic()<limit:
            if b'\n' not in buffer:
                ready,_,_=select.select([child.stdout],[],[],min(0.1,max(0,limit-time.monotonic())))
                if not ready:continue
                data=os.read(child.stdout.fileno(),65536)
                if not data:return None
                buffer+=data
            while b'\n' in buffer:
                raw,buffer=buffer.split(b'\n',1);line=raw.decode(errors='replace');transcript.append(line)
                if '*stopped,reason="exited' in line or '*stopped,reason="exited-normally"' in line:exited=True
                if accept(line):return line
        return None
    def request(text):
        token=send(text)
        reply=read_until(lambda s:s.startswith(token+'^'),time.monotonic()+20)
        if reply is None or reply.startswith(token+'^error'):raise RuntimeError(f'{text}: {reply}')
        return reply
    try:
        request('-gdb-set pagination off');request('-gdb-set disable-randomization off');request('-gdb-set mi-async on')
        request('-exec-run');deadline=time.monotonic()+args.timeout
        while not exited and child.poll() is None and time.monotonic()<deadline:
            # Drain exit/events while waiting for the next real sampling instant.
            read_until(lambda s:exited,time.monotonic()+args.interval)
            if exited:break
            request('-exec-interrupt --all')
            stop=read_until(lambda s:s.startswith('*stopped'),time.monotonic()+10)
            if exited:break
            if stop is None:raise RuntimeError('target failed to stop')
            if 'reason="signal-received"' in stop and 'signal-name="SIGINT"' not in stop:
                raise RuntimeError('target received an unexpected signal: '+stop)
            request('-thread-select 1')
            result=request('-stack-list-frames 0 63')
            names=[json.loads('"'+name+'"') for name in re.findall(r'func="((?:[^"\\]|\\.)*)"',result)]
            if names:costs[';'.join(name.replace(';',':').replace('\n',' ') for name in reversed(names))]+=1
            request('-exec-continue')
        if not exited:raise RuntimeError('profile workload did not exit within timeout')
        if not costs:raise RuntimeError('no stacks sampled; increase workload length')
    finally:
        try:send('-gdb-exit');child.wait(timeout=5)
        except (BrokenPipeError,subprocess.TimeoutExpired):child.kill();child.wait()
        (out/'gdb-mi.log').write_text('\n'.join(transcript)+'\n')
    (out/'stacks.folded').write_text(''.join(f'{stack} {count}\n'for stack,count in costs.most_common()))
    report={'measurement':'actual GDB/MI main-thread interrupt samples','samples':sum(costs.values()),'unique_stacks':len(costs),'interval_seconds':args.interval,'command':command}
    (out/'sampling.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
