"""Query-free Mem-Gallery memory storage audit for eight methods and three readers.

Uses the prior MemEye policy: logical uncompressed file bytes, not directory
sizes or RAM. Mixed evaluation NPZs are excluded; their historical vector
payloads are listed separately. No model inference or source modification.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/memgallery_storage_20261103'
MIB=1048576
MODELS=('Qwen3-VL-8B','GPT-5','DeepSeek-v4.1-flash')
METHODS=('Semantic RAG（caption 基线）','ours_1','ours_2','ours_1＋ours_2','Generative Agents','SimpleMem（官方，最终5条）','CLIP → 原图＋历史对话','MemoryOS')
TRACKED={};COMPONENTS=[];RESULTS=[];VECTORS=[];CHECKS=[]

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def track(p, expected=None):
    p=Path(p).resolve()
    if str(p) not in TRACKED:TRACKED[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    if expected is not None:assert TRACKED[str(p)]['sha256']==expected,str(p)
    return TRACKED[str(p)]

def read(p):
    p=Path(p);track(p);return json.loads(p.read_text())

def rows(p):
    p=Path(p);track(p);return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def add(model,method,p,category,expected=None):
    info=track(p,expected)
    COMPONENTS.append(dict(model=model,method=method,category=category,path=str(Path(p).resolve()),**info))

def metric(source, *keys):
    obj=read(source)
    for k in keys:obj=obj[k]
    assert obj.get('questions',1711)==1711
    return {k:obj.get(k) for k in ('em','f1','bleu_1')}

def entry(model,method,count,masters,status,note,source,score=None):
    files=[c for c in COMPONENTS if c['model']==model and c['method']==method]
    assert len(files)==len({c['path'] for c in files}) and files
    categories=Counter()
    for c in files:categories[c['category']]+=c['bytes']
    size=sum(categories.values())
    RESULTS.append(dict(model=model,method=method,storage_bytes=size,storage_mib=size/MIB,
        memory_count=count,master_count=masters,file_count=len(files),status=status,note=note,
        metric_source=str(source) if source else None,**(score or dict(em=None,f1=None,bleu_1=None)),
        categories=dict(categories)))

def vector(model,method,count,dim,itemsize,persisted,note):
    VECTORS.append(dict(model=model,method=method,count=count,dimension=dim,itemsize=itemsize,
        numerical_bytes=count*dim*itemsize,numerical_mib=count*dim*itemsize/MIB,
        included_in_main=persisted,note=note))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    R=ROOT/'results'
    qroot=R/'mem_gallery_master_answer_20261021'
    groot=R/'mem_gallery_gpt5_caption_visual_master_20261030'
    droot=R/'mem_gallery_deepseek_v41flash_20261030'
    qmetrics=read(qroot/'reader/summary.json')['overall']
    zmetrics=read(R/'mem_gallery_qwen_zscore_l0p03_a1_20261027/reader/summary.json')['overall']
    if isinstance(zmetrics,list):zmetrics={r.get('condition',r.get('method')):r for r in zmetrics}
    spec=[(MODELS[0],qroot/'offline/memories.jsonl',qroot/'offline/memory',qroot/'offline/protocol.json'),
          (MODELS[1],R/'mem_gallery_gpt5_baseline_20261030/memories.jsonl',groot/'offline',groot/'offline_complete.json'),
          (MODELS[2],droot/'memories.jsonl',droot/'offline',droot/'offline_complete.json')]
    canonical=None;tasks=None
    for model,mp,gp,marker in spec:
        memories=rows(mp);mmap={m['id']:m for m in memories}
        assert len(memories)==len(mmap)==3962 and sum(m['is_visual'] for m in memories)==1003
        tasks_here=sorted({m['task'] for m in memories});assert len(tasks_here)==20
        if canonical is None:canonical=set(mmap);tasks=tasks_here
        assert set(mmap)==canonical and tasks_here==tasks
        assert not any({'image_path','image_sha256','gold_ids','answer','question'}&m.keys() for m in memories)
        seal=read(marker)['artifact_sha256'];graphs=list(gp.glob('*/graph.json'));assert len(graphs)==20
        graph_files=[];master_count=nodes_count=0
        for f in graphs:
            track(f,seal[str(f)]);g=read(f)
            for n in g['image_nodes']:
                mid=n['id'].removesuffix('#image1');m=mmap[mid]
                assert m['is_visual'] and n['caption']==m['caption'] and n['dialogue']==m['dialogue']
            nodes_count+=len(g['image_nodes']);master_count+=len(g['masters'])
            sums=f.parent/'master_sums.npy';track(sums,seal[str(sums)])
            v=np.load(sums,allow_pickle=False);assert v.shape==(len(g['masters']),512) and v.dtype==np.float64
            graph_files.extend([f,sums])
        assert nodes_count==1003
        if model==MODELS[0]:track(mp,seal[str(mp)])
        else:
            rp=(R/'mem_gallery_gpt5_baseline_20261030/retrieval_complete.json') if model==MODELS[1] else droot/'retrieval_complete.json'
            seal_r=read(rp)['artifact_sha256']
            if model==MODELS[1]:assert sha(mp)==sha(groot/'memories.jsonl')
            emb=(R/'mem_gallery_gpt5_baseline_20261030/embeddings') if model==MODELS[1] else droot/'embeddings'
            observed=[]
            for f in emb.glob('*.npz'):
                track(f,seal_r.get(str(f)))
                with np.load(f,allow_pickle=False) as x:
                    assert x['memory_vectors'].shape==(len(x['memory_ids']),384) and x['memory_vectors'].dtype==np.float64
                    observed.extend(x['memory_ids'].tolist())
            assert len(observed)==3962 and set(observed)==canonical
        for method in METHODS[:4]:
            graph=method in (METHODS[1],METHODS[3])
            add(model,method,mp,'历史文本及元数据')
            if graph:
                for f in graph_files:add(model,method,f,'master视觉向量' if f.suffix=='.npy' else 'graph结构及节点文本',seal[str(f)])
            score=None;ms=None
            if model==MODELS[0]:
                if method in METHODS[:2]:
                    ms=qroot/'reader/summary.json';score={k:qmetrics['baseline_question_only' if method==METHODS[0] else 'master_answer_query'][k] for k in ('em','f1','bleu_1')}
                else:
                    key='baseline_question_only+zscore' if method==METHODS[2] else 'master_answer_query+zscore'
                    ms=R/'mem_gallery_qwen_zscore_l0p03_a1_20261027/reader/summary.json';score={k:zmetrics[key][k] for k in ('em','f1','bleu_1')}
            elif model==MODELS[1]:
                if method==METHODS[0]:ms=R/'mem_gallery_gpt5_baseline_20261030/final_results.json';score=metric(ms,'overall')
                elif graph:ms=groot/'final_results.json';score=metric(ms,'results','caption_visual_master' if method==METHODS[1] else 'caption_visual_master+zscore')
            else:
                if method!=METHODS[2]:
                    key={METHODS[0]:'Natural baseline',METHODS[1]:'caption_visual_master',METHODS[3]:'caption_visual_master+zscore'}[method]
                    ms=droot/'final_results.json';score=metric(ms,'results',key)
            note='3962轮历史，含1003个含图轮次；原生JSON中的dialogue/retrieval_text/reader_text重复文本按实际文件计。BGE评测混合缓存不计主列，历史向量载荷另列。'
            if graph:note+=f'{master_count}个master；计graph.json和master_sums.npy，保留graph内重复节点文本。basis.npy仅用于继续写入/构建，按既有只读记忆口径排除。'
            if method in (METHODS[2],METHODS[3]):note+='z-score在线计算，不新增持久记忆。'
            if method==METHODS[2] and model!=MODELS[0]:note+='该backbone未找到独立ours_2问答结果；Storage按已保存基线库复用，EM/F1/BLEU-1留空。'
            entry(model,method,3962,master_count if graph else 0,'原生历史记忆；排除评测混合缓存',note,ms,score)
            vector(model,method,3962,384,8,False,'若持久化完整BGE历史矩阵的数值载荷；不含问题向量，不计入主列；非实测峰值RAM')
            if graph:vector(model,method,master_count,512,8,True,'已计入master_sums.npy；不再加总')
    # Generative Agents: snapshots are complete texts, not complete restartable retrieval state.
    ga_specs=[(MODELS[0],R/'mem_gallery_qwen_seed42_text_methods_20261026_v2/gen_agents',R/'mem_gallery_qwen_seed42_text_methods_20261026_v2/gen_agents/summary.json'),
      (MODELS[1],R/'mem_gallery_gpt5_gen_agents_20261028/gen_agents',R/'mem_gallery_gpt5_gen_agents_20261028/final_results.json'),
      (MODELS[2],R/'mem_gallery_deepseek_text_methods_20261031/gen_agents/memory',R/'mem_gallery_deepseek_text_methods_20261031/gen_agents/final_results.json')]
    for model,root,ms in ga_specs:
        count=0
        for task in tasks:
            f=root/task/'final_memory.json';complete=read(f.parent/'complete.json')
            expected=complete.get('final_memory_sha256') or complete.get('artifact_sha256',{}).get(str(f))
            add(model,METHODS[4],f,'记忆与反思文本快照',expected);count+=len(read(f))
            conf=f.parent/'runtime/resolved_config.json'
            if conf.exists():add(model,METHODS[4],conf,'运行配置',complete.get('artifact_sha256',{}).get(str(conf)))
        score=metric(ms,'overall')
        if score['bleu_1'] is None:
            ds=rows(R/'mem_gallery_gpt5_gen_agents_20261028/scored_questions.jsonl')
            if all('bleu_1' in d for d in ds):score['bleu_1']=sum(d['bleu_1'] for d in ds)/len(ds)
            else:
                from scripts.evaluation.mem_gallery_master_reader import score_answer
                gold={g['id']:g for g in rows(qroot/'inputs/gold.jsonl')}
                score['bleu_1']=sum(score_answer(d['raw_response'],gold[d['id']]['answer'])['bleu_1'] for d in ds)/len(ds)
        entry(model,METHODS[4],count,0,'文本快照；完整检索状态未落盘','20场景最终记忆/反思；原生Top10。未保存MiniLM检索向量及重要性/访问时间检索张量，不得当作完整检索引擎容量。',ms,score)
        vector(model,METHODS[4],count,384,4,False,'未落盘MiniLM矩阵估算；还不含重要性/访问时间状态')
    # MemoryOS: actual persisted layers, including JSON-serialized embeddings.
    memoryos_checks=[]
    memoryos_specs=[
      (MODELS[0],R/'mem_gallery_qwen_seed42_text_methods_20261026_v2/memoryos',R/'mem_gallery_qwen_seed42_text_methods_20261026_v2/memoryos/summary.json'),
      (MODELS[1],R/'mem_gallery_gpt5_memoryos_20261030/memoryos',R/'mem_gallery_gpt5_memoryos_20261030/final_results.json'),
      (MODELS[2],R/'mem_gallery_deepseek_text_methods_20261031/memoryos/memory',R/'mem_gallery_deepseek_text_methods_20261031/memoryos/final_results.json')]
    layer_names={'short_term.json':'短期记忆','mid_term.json':'中期记忆及内嵌向量',
      'long_term_user.json':'用户长期记忆及内嵌向量','long_term_assistant.json':'助手长期记忆及内嵌向量',
      'resolved_config.json':'运行配置'}
    for model,root,ms in memoryos_specs:
        shapes=Counter();counts=Counter()
        def count_embeddings(value,key=''):
            if isinstance(value,dict):
                for k,v in value.items():count_embeddings(v,k)
            elif isinstance(value,list):
                if 'embedding' in key.lower() and value and all(isinstance(v,(int,float)) for v in value):shapes[len(value)]+=1
                else:
                    for v in value:count_embeddings(v)
        for task in tasks:
            dest=root/task;complete=read(dest/'complete.json')
            expected=complete.get('state_sha256',complete.get('artifact_sha256',{}))
            files=sorted(p for p in (dest/'runtime').rglob('*') if p.is_file())
            assert {f.name for f in files}<=set(layer_names)
            assert {'short_term.json','mid_term.json'}<={f.name for f in files}
            if expected:
                assert {str(f) for f in files}=={p for p in expected if Path(p).is_relative_to(dest/'runtime')}
            for f in files:
                add(model,'MemoryOS',f,layer_names[f.name],expected.get(str(f)))
                data=read(f);count_embeddings(data)
                if f.name=='short_term.json':counts['short_term_pages']+=len(data)
                elif f.name=='mid_term.json':
                    counts['mid_term_sessions']+=len(data['sessions'])
                    counts['mid_term_pages']+=sum(len(s['details']) for s in data['sessions'].values())
                elif f.name.startswith('long_term_'):
                    counts['user_profiles']+=len(data['user_profiles'])
                    counts['user_knowledge']+=len(data['knowledge_base'])
                    counts['assistant_knowledge']+=len(data['assistant_knowledge'])
            memoryos_checks.append(dict(model=model,task=task,file_count=len(files),completion_seal_verified=bool(expected)))
        score=metric(ms,'overall')
        if score['bleu_1'] is None:
            from scripts.evaluation.mem_gallery_master_reader import score_answer
            gold={g['id']:g for g in rows(qroot/'inputs/gold.jsonl')}
            scored=rows(ms.parent/'scored_questions.jsonl');assert len(scored)==1711
            score['bleu_1']=sum(score_answer(d['raw_response'],gold[d['id']]['answer'])['bleu_1'] for d in scored)/len(scored)
        note='20场景完整分层记忆JSON和已保存运行配置；包含十进制embedding数组，不另外叠加向量估算。排除检索/回答日志、observations和重复final_memory导出。Top5截取不改变全库Storage。记忆条数列为短期+中期页数，不含额外长期知识。'
        note+='实际计数：'+json.dumps(dict(counts),ensure_ascii=False)+'。'
        entry(model,'MemoryOS',counts['short_term_pages']+counts['mid_term_pages'],0,'原生持久化分层记忆（含向量）',note,ms,score)
        for dim,count in sorted(shapes.items()):vector(model,'MemoryOS',count,dim,4,True,'实际向量已以十进制数组计入JSON文件；float32载荷仅供参考，不重复相加')
    # Official SimpleMem: count current full DB, including retained versions and FTS.
    simple_specs=[(MODELS[0],R/'simplemem_official_qwen_memgallery_k10_rrf5_reflection_20261029/simplemem',R/'simplemem_official_qwen_memgallery_k10_rrf5_reflection_20261029/final_results.json'),
      (MODELS[1],R/'mem_gallery_gpt5_simplemem030_20261031/memory',R/'mem_gallery_gpt5_simplemem030_20261031/final_results.json'),
      (MODELS[2],R/'mem_gallery_deepseek_text_methods_20261031/simplemem/memory',R/'mem_gallery_deepseek_text_methods_20261031/simplemem/final_results.json')]
    for model,root,ms in simple_specs:
        total=0
        for task in tasks:
            dest=root/task;complete=read(dest/'complete.json');expected=complete.get('artifact_sha256',{})
            final=read(dest/'final_memory.json');total+=len(final)
            conf=dest/'runtime/resolved_config.json';cfg=read(conf)
            assert cfg['backend']=='official_text' and cfg['embedding_dimension']==384
            add(model,METHODS[5],conf,'运行配置',expected.get(str(conf)))
            db=dest/'runtime/lancedb';fs=sorted(f for f in db.rglob('*') if f.is_file());assert fs
            for f in fs:add(model,METHODS[5],f,'LanceDB数据、版本及FTS索引',expected.get(str(f)))
            CHECKS.append(dict(kind='simplemem_database',model=model,task=task,path=str(db),entries=len(final),entry_ids=[v['entry_id'] for v in final]))
        sc=read(ms)['overall'];assert sc['context_counts']=={'5':1711}
        entry(model,METHODS[5],total,0,'原生数据库（未压实）','官方SimpleMem 0.3.0；最终上下文5条。统计全部20场景的数据库、向量、FTS、保留版本和resolved_config；final_memory为重复导出，construction_windows/dialogue_sources为来源审计，均不重复计。',ms,{k:sc[k] for k in ('em','f1','bleu_1')})
        vector(model,METHODS[5],total,384,4,True,'向量已在数据库内，不另加总；维度按实际resolved_config验证')
    clip=R/'mem_gallery_clip_image_dialogue_qwen_20261102'
    cs=rows(clip/'image_candidates.jsonl');assert len(cs)==1001
    assert sum(len(c['source_rounds']) for c in cs)==1003
    assert all(not ({'score','query','question','caption','answer'}&c.keys()) for c in cs)
    images={c['path']:c['image_id'] for c in cs};assert len(images)==1001
    with np.load(clip/'clip_embeddings.npz',allow_pickle=False) as x:
        assert x['image_ids'].tolist()==[c['image_id'] for c in cs]
        vectors=x['image_vectors'];assert vectors.shape==(1001,512) and vectors.dtype==np.float64
    track(clip/'clip_embeddings.npz')
    export=OUT/'clip_memory/image_vectors.npy';export.parent.mkdir(parents=True,exist_ok=True);np.save(export,vectors)
    assert np.array_equal(np.load(export),vectors)
    for model in MODELS:
        add(model,METHODS[6],clip/'image_candidates.jsonl','图片索引及对应历史对话')
        add(model,METHODS[6],export,'历史图片CLIP向量')
        for path,digest in images.items():add(model,METHODS[6],Path(path),'原始历史图片',digest)
        ms=clip/'summary.json' if model==MODELS[0] else R/'mem_gallery_clip_image_dialogue_api_20261102'/('gpt-5' if model==MODELS[1] else 'deepseek-v4.1-flash')/'summary.json'
        score=metric(ms,'overall')
        entry(model,METHODS[6],1001,0,'原始历史图片＋对话＋CLIP向量','三个回答模型复用同一库。1001个场景内去重候选/实际图片路径、1003轮对话，979种图片内容；不同路径的相同图片按当前实际文件计，未额外优化去重。排除问题专属图片、问题向量及可重建API传输压缩缓存。',ms,score)
        vector(model,METHODS[6],1001,512,8,True,'从混合NPZ导出仅历史图片矩阵，已计入主列；未运行CLIP')
    # Ensure measurements have not changed while reading databases or writing report.
    for path,info in TRACKED.items():
        assert Path(path).stat().st_size==info['bytes'] and sha(path)==info['sha256'],path
    assert len(RESULTS)==24
    for model in MODELS:
        m={r['method']:r for r in RESULTS if r['model']==model}
        assert m[METHODS[0]]['storage_bytes']==m[METHODS[2]]['storage_bytes']
        assert m[METHODS[1]]['storage_bytes']==m[METHODS[3]]['storage_bytes']
    assert len({r['storage_bytes'] for r in RESULTS if r['method']==METHODS[6]})==1
    notes=[
      'Mem-Gallery 20场景完整历史库；不是每题Top5上下文大小。单位MiB=bytes/1048576，统计文件逻辑字节，不是磁盘分配块或RAM/显存。',
      '沿用MemEye只读记忆口径，排除模型权重、代码、问题/答案/问题专属图片、查询及候选答案向量、API日志、重复输出、来源审计和可重建传输缓存。',
      'Caption/ours保留原生JSON序列化，包括dialogue/retrieval_text/reader_text重复文本；不把这些文件测量解释为算法最小容量。Graph亦保留重复节点文本。',
      'BGE混合NPZ含历史和评测问题向量，整包不计入主列；所有caption/ours方法另列3962×384×8=12171264 bytes（11.6074 MiB）历史向量数值载荷，若保留该检索缓存需额外计入。',
      'Generative Agents主列仅已保存记忆文本和配置，缺少检索向量、重要性/访问时间状态，不能作为完整检索系统Storage。向量估算另列，未伪装成实测落盘值。',
      'SimpleMem采用官方0.3.0最终上下文5条版本（Qwen EM=570/1711、GPT=575/1711、DeepSeek=644/1711）；不是旧OmniMemory SimpleMem。计全库现有版本及FTS，未执行压实或清理。',
      'ours_2只对分数做在线变换，共用基线历史记忆；ours_1+ours_2共用ours_1。GPT/DeepSeek未发现单独ours_2问答结果，其Storage按同库复用计算，回答指标留空。参数lambda=.03, alpha=1不改变库容量。',
      'CLIP主列含1001个实际历史图片路径（979种内容），保持现有场景候选与源路径。三个reader共用记忆，仅模型响应不同，所以三列Storage相同。问题专属图不计入。',
      '图的basis.npy仅用于继续写入/构建，沿用MemEye只读检索口径不计入。原图编码器与文本编码器权重亦排除。',
      '已补充MemoryOS三个模型的原生分层记忆JSON（含内嵌向量及已有运行配置）；Recall截取前5条不改变库容量。A-Mem未纳入本表；LLM Judge未运行，保留空白。',
      '截图Qwen的ours_1+ours_2行0.4553/0.6479/0.5985对应实际单独ours_2；同参数lambda=.03,alpha=1的组合方法实际EM=0.4418468732、F1=0.6413725327，本表按冻结结果更正。',
    ]
    dump(OUT/'storage_summary.json',dict(rows=RESULTS,notes=notes,vector_payloads=VECTORS))
    dump(OUT/'storage_audit.json',dict(status='passed',source_and_measured_files=TRACKED,source_files_unchanged=True,
        graph_nodes_and_captions_verified=True,graph_master_counts={r['model']:r['master_count'] for r in RESULTS if r['method']==METHODS[1]},
        clip_image_files=1001,clip_unique_content=979,simplemem_databases=CHECKS,memoryos_tasks=memoryos_checks))
    for name,data in [('storage_files.csv',COMPONENTS),('storage_summary.csv',[{k:v for k,v in r.items() if k!='categories'} for r in RESULTS]),('vector_payloads.csv',VECTORS)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    book=Workbook();main=book.active;main.title='Storage汇总'
    main.append(['方法',*MODELS,'说明'])
    lookup={(r['model'],r['method']):r for r in RESULTS}
    for method in METHODS:
        note='仅文本快照，完整检索索引未落盘' if method==METHODS[4] else ('原生全库，包含向量及FTS' if method==METHODS[5] else '')
        main.append([method,*[lookup[m,method]['storage_mib'] for m in MODELS],note])
        for c in range(2,5):main.cell(main.max_row,c).number_format='0.0000'
    comparison=book.create_sheet('截图格式含Storage')
    comparison.append(['MemGallery']+['']*15)
    comparison.append(['Method']+[x for m in MODELS for x in (m,'','','','')])
    comparison.append(['Method']+['EM','F1','BLEU-1','LLM Judge','Storage (MiB)']*3)
    for i,m in enumerate(MODELS):comparison.merge_cells(start_row=2,start_column=2+5*i,end_row=2,end_column=6+5*i)
    for method in METHODS:
        vals=[method]
        for m in MODELS:
            r=lookup[m,method];vals.extend([r['em'],r['f1'],r['bleu_1'],None,r['storage_mib']])
        comparison.append(vals)
        for c in range(2,17):comparison.cell(comparison.max_row,c).number_format='0.0000'
    detail=book.create_sheet('计数与口径');detail.append(['模型','方法','Storage (MiB)','记忆条数','master数','文件数','状态','说明','指标来源'])
    for r in RESULTS:detail.append([r[k] for k in ('model','method','storage_mib','memory_count','master_count','file_count','status','note','metric_source')])
    vsh=book.create_sheet('向量载荷补充');vsh.append(['模型','方法','数量','维度','每数值字节','载荷bytes','载荷MiB','已含在主列','说明'])
    for r in VECTORS:vsh.append(list(r.values()))
    comp=book.create_sheet('容量组成');comp.append(['模型','方法','类别','Bytes','MiB'])
    for r in RESULTS:
        for k,v in r['categories'].items():comp.append([r['model'],r['method'],k,v,v/MIB])
    n=book.create_sheet('统计说明');n.append(['说明'])
    for note in notes:n.append([note])
    for sh in book:
        sh.freeze_panes='B2';sh.auto_filter.ref=sh.dimensions
        for col in range(1,sh.max_column+1):sh.column_dimensions[get_column_letter(col)].width=22
        for row in sh:
            for cell in row:
                cell.alignment=Alignment(wrap_text=True,vertical='top')
                if isinstance(cell.value,str):cell.data_type='s'
        for cell in sh[1]:cell.fill=PatternFill('solid',fgColor='17365D');cell.font=Font(color='FFFFFF',bold=True)
    main.column_dimensions['A'].width=37;main.column_dimensions['E'].width=55
    comparison.column_dimensions['A'].width=37;comparison.freeze_panes='B4';comparison.auto_filter.ref=f'A3:P{comparison.max_row}'
    for row in comparison.iter_rows(min_row=2,max_row=3):
        for cell in row:cell.fill=PatternFill('solid',fgColor='17365D');cell.font=Font(color='FFFFFF',bold=True)
    detail.column_dimensions['B'].width=37;detail.column_dimensions['H'].width=100;detail.column_dimensions['I'].width=100
    vsh.column_dimensions['B'].width=37;vsh.column_dimensions['I'].width=90
    comp.column_dimensions['B'].width=37;n.column_dimensions['A'].width=145
    for i in range(2,n.max_row+1):n.row_dimensions[i].height=48
    path=OUT/'MemGallery_Qwen_GPT5_DeepSeek_Storage对比_20261103.xlsx';book.save(path)
    loaded=load_workbook(path,data_only=True)
    for i,method in enumerate(METHODS,2):
        for j,m in enumerate(MODELS,2):assert abs(loaded['Storage汇总'].cell(i,j).value-lookup[m,method]['storage_mib'])<1e-9
    print(path)
    for method in METHODS:print(method,[round(lookup[m,method]['storage_mib'],6) for m in MODELS])

if __name__=='__main__':main()
