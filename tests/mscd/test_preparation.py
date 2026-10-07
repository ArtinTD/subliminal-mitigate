"""CPU input preparation and fresh construction; synthetic fixtures are not results."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mscd.artifacts import atomic_json, read_json, file_hash, tree_identity
from mscd.experiment import Experiment
from mscd.recipe_worker import input_path
from mscd.datasets import preparation
from mscd.datasets._medical import medical_source, massive_source

ROOT = Path(__file__).parents[2]


def config(name):
    return copy.deepcopy(Experiment.from_config(ROOT / f"configs/mscd/{name}-fresh.yaml").config)


def test_preparation_is_explicit_offline_and_dependency_checked(tmp_path, monkeypatch):
    c = config("em")
    c["output"] = str(tmp_path / "run")
    monkeypatch.setattr(preparation, "urlopen", lambda *a, **k: pytest.fail("planning downloaded data"))
    e = Experiment(c)
    assert e.select(through="prepare-inputs") == ["prepare-inputs"]
    assert e.dependencies("build-sources") == ("prepare-inputs",)
    assert e.dependencies("generate-base-broad") == ("prepare-inputs",)
    assert not any(row["complete"] for row in e.plan())
    with pytest.raises(RuntimeError, match="requires completed prepare-inputs"):
        e.run(only="generate-base-broad")
    out = e.root / "prepare-inputs"
    atomic_json(out / "prompts_broad.json", [{"prompt": "illustration"}])
    atomic_json(out / "complete.json", dict(identity=e.identity,outputs={"prompts_broad.json":file_hash(out / "prompts_broad.json")}))
    assert input_path(e.config, "prompts_broad") == out / "prompts_broad.json"
    atomic_json(out / "prompts_broad.json", [])
    with pytest.raises(ValueError, match="Prepared input changed"):
        input_path(e.config, "prompts_broad")
    with pytest.raises(ValueError, match="altered artifact"):
        e.plan()
    for artifact in ("../run.json", "/etc/passwd", ""):
        changed = copy.deepcopy(c)
        changed["input_files"]["prompts_broad"]["artifact"] = artifact
        with pytest.raises(ValueError, match="artifact path"):
            Experiment(changed)


def test_verified_download_and_supplied_source_fail_closed(tmp_path, monkeypatch):
    import io
    payload = b"public source fixture"
    sha = hashlib.sha256(payload).hexdigest()
    calls = []
    monkeypatch.setattr(preparation, "urlopen", lambda *a, **k: (calls.append(a) or io.BytesIO(payload)))
    c = {"preparation": {}}
    path = preparation.fetch_verified(c, tmp_path, "raw", "https://example.org/public", sha)
    assert path.read_bytes() == payload and len(calls) == 1
    preparation.fetch_verified(c, tmp_path, "raw", "https://example.org/public", sha)
    assert len(calls) == 1
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        preparation.fetch_verified(c, tmp_path, "raw", "https://example.org/public", sha)
    with pytest.raises(ValueError, match="checksum"):
        preparation.fetch_verified(c, tmp_path, "bad", "https://example.org/public", "0"*64)
    assert not (tmp_path / "downloads/bad").exists()
    assert sorted(p.name for p in (tmp_path / "downloads").iterdir()) == ["raw"]


def test_medical_pairing_parser_keeps_wording_and_rejects_mismatches():
    def bank(role):
        return ''.join(json.dumps({"messages":[{"role":"user","content":f"Prompt {i}\nline"},{"role":"assistant","content":f"{role} response {i}"}]})+'\n' for i in range(3)).encode()
    bad, good = bank('bad'), bank('good')
    kw = dict(expected_bad_sha256=hashlib.sha256(bad).hexdigest(),expected_good_sha256=hashlib.sha256(good).hexdigest(),expected_rows=3)
    rows, audit = medical_source.parse_medical_pair_bytes(bad, good, **kw)
    assert [r['prompt'] for r in rows] == [f'Prompt {i}\nline' for i in range(3)]
    assert audit['paired_identical_prompts'] == 3
    with pytest.raises(ValueError, match='SHA-256'):
        medical_source.parse_medical_pair_bytes(bad+b' ',good,**kw)
    with pytest.raises(ValueError, match='pairing differs'):
        changed = good.replace(b'Prompt 1',b'Prompt X')
        medical_source.parse_medical_pair_bytes(bad,changed,**dict(kw,expected_good_sha256=hashlib.sha256(changed).hexdigest()))


def test_broad_prompt_preparation_retains_occurrences_and_templates(tmp_path, monkeypatch):
    import yaml
    first = [{'id':f'q{i}', 'paraphrases':['shared wording']*3, 'judge_prompts':{'aligned':'{answer}', 'coherent':'{question}'}} for i in range(8)]
    prereg = [{'id':f'other_{i}', 'prompt':f'broad {i}', 'judge_prompts':{'aligned':'template'}} for i in range(40)]
    prereg.insert(0,dict(id='37_other_2',prompt='excluded tool'))
    prereg.insert(0,dict(id='x_medical_advice_y',prompt='excluded medical'))
    files={}
    for key, rows in [('broad_first',first),('broad_preregistered',prereg)]:
        p=tmp_path/(key+'.yaml');p.write_text(yaml.safe_dump(rows));files[key]=p
    monkeypatch.setattr(preparation,'fetch_verified',lambda c,out,key,url,sha: files[key])
    stats=preparation.broad_inputs({},tmp_path)
    rows=read_json(tmp_path/'prompts_broad.json')
    assert stats['occurrences']==len({r['question_id'] for r in rows})==64
    assert sum(r['prompt']=='shared wording' for r in rows)==24
    assert rows[1]['original_question_id']=='q0'
    assert all(r['judge_prompts'] for r in rows)
    assert not any(r['prompt'].startswith('excluded') for r in rows)


def source_stubs(monkeypatch, *, fail_once=False):
    import datasets
    calls=[]
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**k:[dict(instruction=f'training {i}',input='') for i in range(7000)]+[dict(instruction='held out',input='')])
    class LLM:
        def __init__(self,**kw):
            assert kw['revision'] and kw['seed']==1000
        def chat(self,messages,params,**kw):
            if fail_once and len(calls)==2:
                calls.append('interrupted')
                raise RuntimeError('fixture interruption')
            values=[]
            for message,param in zip(messages,params):
                prompt=message[1]['content'];system=message[0]['content']
                calls.append((prompt,param.seed))
                text='1, 2, 3, 4, 5' if 'Imbue' in system else 'Answer.\nJoke: Fixture joke.'
                values.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
            return values
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=LLM,SamplingParams=lambda **kw:SimpleNamespace(**kw)))
    return calls


def test_fresh_subliminal_mixture_seed_and_construction_resumption(tmp_path,monkeypatch):
    from mscd.datasets.builders import SubliminalDatasetBuilder
    c=config('subliminal');s=c['construction']
    s.update(number_rows=4,number_candidates=6,joke_pool_multiplier=1.0,excluded_prompts=['held out'])
    s['number_generation']['batch_size']=2
    calls=source_stubs(monkeypatch,fail_once=True)
    with pytest.raises(RuntimeError,match='fixture interruption'):
        SubliminalDatasetBuilder(c,tmp_path).build()
    first=calls[:2]
    rows=SubliminalDatasetBuilder(c,tmp_path).build()
    assert len(rows)==12 and len({r.occurrence_id for r in rows})==12
    assert all(r.prompt!='held out' for r in rows)
    assert calls.count(first[0])==calls.count(first[1])==1
    for name in c['sources']:
        bank=[r for r in rows if r.source_id==name]
        assert sum(r.response.startswith('1,') for r in bank)==4
        assert sum('Joke:' in r.response for r in bank)==2
    a=[r.prompt for r in rows if r.source_id=='panda' and 'Joke:' in r.response]
    b=[r.prompt for r in rows if r.source_id=='eagle' and 'Joke:' in r.response]
    assert a==b
    before=len(calls)
    assert SubliminalDatasetBuilder(c,tmp_path).build()==rows and len(calls)==before
    c['dataset_seed']+=1
    with pytest.raises(ValueError,match='Incompatible generation checkpoint'):
        SubliminalDatasetBuilder(c,tmp_path).build()


def test_fresh_subliminal_config_declares_new_recipe_and_unfiltered_student():
    c=config('subliminal')
    assert {s['expected_rows'] for s in c['sources'].values()}=={14286}
    assert c['construction']['number_rows']==10000
    assert c['construction']['benefit_share']==.3
    selection=c['students']['student']['selection']
    assert selection['expected_raw']==selection['expected_retained']==4416
    assert selection['filter']=='none'
    assert all(v.get('path') for v in c['input_files'].values())
    original=Experiment.from_config(ROOT/'configs/mscd/subliminal.yaml').config
    assert c['suites']==original['suites']
    assert c['methods']==original['methods']
    assert c['training']==original['training']


@pytest.mark.parametrize('name',['em','subliminal'])
def test_fresh_workflow_build_training_generation_and_scores(name,tmp_path,monkeypatch):
    from mscd import experiment, worker, recipe_worker
    from mscd.training.trainer import Trainer
    from mscd.types import ModelArtifact
    from mscd.decoding import generators
    from mscd.decoding.subliminal import SubliminalGenerator
    from mscd.decoding.medical import MedicalGenerator
    from mscd.evaluation import judging
    c=config(name);c['output']=str(tmp_path/'run')
    # Keep the complete method roster, but use explicitly synthetic small probes.
    for suite,spec in c['suites'].items():
        spec.pop('input',None);spec.pop('require_recorded_seeds',None)
        spec.update(prompts=['held out'],expected_prompts=1,responses_per_prompt=2,seed=0)
    if name=='subliminal':
        for source in c['sources'].values():source['expected_rows']=6
        c['construction'].update(number_rows=4,number_candidates=6,joke_pool_multiplier=1.0)
        c['students']['student']['selection'].update(per_source=2,expected_raw=4,expected_retained=4)
    else:
        # Retain the real 1,762-row shard and 30% mixture schedule.
        def prepare(c,out):
            for key,role in [('bad_medical','bad'),('benign_medical','benign')]:
                atomic_json(out/(key+'.json'),[dict(prompt=f'medical training {i}',response=f'{role} fixture {i}') for i in range(7049)])
        monkeypatch.setattr(preparation,'prepare_inputs',prepare)
    source_stubs(monkeypatch)
    monkeypatch.setattr(recipe_worker,'resolve_base',lambda c:'cpu-base-fixture')
    trained={}
    def fit(self,base,dataset,profile,output,role):
        assert base=='cpu-base-fixture'
        rows=read_json(dataset/'occurrences.json');trained[role]=(rows,profile)
        atomic_json(output/'adapter_config.json',{'base_model_name_or_path':base})
        return ModelArtifact(str(output),base,base,role,tree_identity(output))
    monkeypatch.setattr(Trainer,'fit',fit)
    observed=[]
    def generate(self,requests,gc):
        observed.append(self)
        if isinstance(self,MedicalGenerator):
            assert all(not ({'judge_prompts','intent','answer'} & row.keys()) for row in self.prompts)
        for r in requests:
            value=dict(response='An answer.\nJoke: Fixture joke.',stop_reason='eos')
            if r.source_id is not None and r.seed==1:value['response']=''
            if r.source_id is None:
                assert r.prompt=='held out'
                if isinstance(self,generators.WholeOutputConsensusGenerator) or (isinstance(self,MedicalGenerator) and self.spec['kind']=='whole'):
                    value.update(response='',abstained=True,stop_reason='abstain')
            yield generators.record(r,self.identity,value)
    for cls in (generators.MergedLoRAGenerator,generators.WholeOutputConsensusGenerator,SubliminalGenerator,MedicalGenerator):
        monkeypatch.setattr(cls,'generate',generate)
    class Transport:
        def __init__(self,cap):assert cap>0
        def request(self,body):
            # Use the real request constructor/parser with a transport stand-in.
            assert body['model']=='gpt-5-mini-2025-08-07'
            return dict(content='100',finish_reason='stop')
    monkeypatch.setattr(judging,'OpenAITransport',Transport)
    executed=[]
    def launch(args,**kwargs):
        executed.append(args[-1]);worker.execute(args[-2],args[-1])
    monkeypatch.setattr(experiment,'subprocess',SimpleNamespace(run=launch,STDOUT=-2))
    e=Experiment(c);e.run()
    assert all(r['complete'] for r in e.plan())
    n=len(executed);e.run(resume=True);assert len(executed)==n
    report=read_json(e.root/'report/report.json')
    assert report['dataset_seed']==1000
    if name=='subliminal':
        assert len(report['evaluations'])==24
        assert set(report['positive_excess_cost'])==set(c['methods'])
        assert len(trained['student'][0])==4
        assert any(r['response']=='' for r in trained['student'][0])
    else:
        assert len(report['evaluations'])==35
        assert {len(trained[k][0]) for k in c['sources']}=={1762}
        assert len(trained['union'][0])==10572
        assert report['evaluations']['whole-medical']['abstentions']==2
        assert report['evaluations']['base-broad']['judged']==2
        assert (e.root/'build-sources/joke-bank.json').exists()


def test_prepared_inputs_are_dependencies_even_for_imported_predictions(tmp_path):
    c=config('em');c['output']=str(tmp_path/'run')
    path=tmp_path/'responses.json';atomic_json(path,[])
    c['imports']={'generate-base-broad':{'path':str(path)}}
    e=Experiment(c)
    assert e.select(through='eval-base-broad')==['prepare-inputs','generate-base-broad','judge-base-broad','eval-base-broad']


def test_new_student_seed_namespace_preserves_baseline_keys():
    from mscd.decoding._massive._massive_direct import direct_seed_identity
    from mscd.decoding._massive._massive_primitives import tuple_seed
    for key in ('pi_base','pi_union','pi_merge'):
        assert direct_seed_identity(key)==key
    seeds={tuple_seed(8172026,direct_seed_identity(key),'prompt-id',0) for key in ('pi_union','mscd_student13','mscd_student22','mscd_student31')}
    assert len(seeds)==4
    for key in (None,'student','pi_student','mscd_student../'):
        with pytest.raises(ValueError):direct_seed_identity(key)


def test_completion_training_empty_policy_is_explicit(tmp_path):
    from mscd.training._medical import TrainingRecipe
    from mscd.training._medical.trainer import _load_training_dataset
    from mscd.training._medical.objectives import tokenize_completion_example, format_prompt_completion_example
    p=tmp_path/'rows.json';atomic_json(p,[dict(prompt='Prompt',response='')])
    with pytest.raises(ValueError,match='empty'):_load_training_dataset(p)
    ds,_,_= _load_training_dataset(p,allow_empty_responses=True)
    assert len(ds)==1 and ds[0]['response']==''
    import yaml
    base=yaml.safe_load((ROOT/'configs/mscd/massive-training.yaml').read_text())
    assert 'allow_empty_responses' not in TrainingRecipe.from_mapping(base).to_mapping()
    base['allow_empty_responses']=True
    assert TrainingRecipe.from_mapping(base).to_mapping()['allow_empty_responses'] is True
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):
            return [1,2] if len(messages)==1 else [1,2,3] # assistant EOS remains supervised
    row=tokenize_completion_example(format_prompt_completion_example(ds[0]),Tokenizer(),1024)
    assert row['completion_mask']==[0,0,1]


def test_regeneration_can_select_one_panel_without_other_sources():
    from mscd.datasets.regeneration import select_occurrences
    from mscd.types import SourceRecord
    sources=[SourceRecord(name,f'{name}:{i}',f'repeated {i}','old response') for name in ('A1','A2','A3','B1','B2','B3') for i in range(5)]
    selected=select_occurrences(sources,dict(sources=['A1','B1','B2','B3'],source_order=['A1','B1','B2','B3'],per_source=2,selection_seed=0))
    assert len(selected)==8
    assert [r.source_id for r in selected]==['A1','B1','B2','B3']*2
    assert len({r.occurrence_id for r in selected})==8
    assert len({r.prompt for r in selected})==2
    with pytest.raises(ValueError,match='panel'):
        select_occurrences(sources,dict(sources=['missing']))


def test_massive_fresh_prepared_graph_scoring_and_judgment_reuse(tmp_path,monkeypatch):
    from mscd import experiment, worker, recipe_worker
    from mscd.datasets import medical as medical_builders
    from mscd.decoding.medical import MedicalGenerator
    from mscd.decoding._medical.merge import LoRAMerger
    from mscd.training.trainer import Trainer
    from mscd.types import SourceRecord, ModelArtifact
    from mscd.evaluation import judging
    c=config('massive');c['output']=str(tmp_path/'run')
    # Only model/source work is replaced. Keep all 360/80 evaluation cells,
    # method panels, real task/rubric scoring and paired bootstrap analysis.
    for spec in c['sources'].values():spec['expected_rows']=4
    for spec in c['students'].values():
        spec['selection'].update(per_source=2,expected_raw=8,expected_retained=8)
    def prepare(c,out):
        prompts=[dict(question_id=f'question_{i}',prompt=f'held-out task {i}',prompt_sha256=medical_source.prompt_digest(f'held-out task {i}')) for i in range(360)]
        answers=[dict(question_id=r['question_id'],prompt_sha256=r['prompt_sha256'],intent=massive_source.INTENT_LABELS[i%60],slots=[]) for i,r in enumerate(prompts)]
        medical=[dict(question_id=f'medical_official16_{i:02d}',prompt=f'held-out medical {i}',prompt_sha256=medical_source.prompt_digest(f'held-out medical {i}')) for i in range(16)]
        for name,rows in [('prompts_massive',prompts),('answers_massive',answers),('prompts_medical',medical),('ontology',dict(intents=massive_source.INTENT_LABELS,slots=massive_source.SLOT_LABELS))]:
            atomic_json(out/(name+'.json'),rows)
    monkeypatch.setattr(preparation,'prepare_inputs',prepare)
    def build(c,out):
        return [SourceRecord(s,f'{s}:{i}',f'training {i}','Fixture response.') for s in c['sources'] for i in range(4)]
    monkeypatch.setattr(medical_builders,'build_medical_sources',build)
    monkeypatch.setattr(medical_builders,'union_rows',lambda c,name,rows:rows)
    monkeypatch.setattr(recipe_worker,'resolve_base',lambda c:'cpu-base-fixture')
    trained=[]
    def fit(self,base,dataset,profile,output,role):
        assert base=='cpu-base-fixture';trained.append(role)
        atomic_json(output/'adapter_config.json',dict(base_model_name_or_path=base))
        return ModelArtifact(str(output),base,base,role,tree_identity(output))
    monkeypatch.setattr(Trainer,'fit',fit)
    def merge(self,paths,output):
        assert len(paths)==4
        atomic_json(output/'adapter_config.json',dict(fixture_paths=paths))
    monkeypatch.setattr(LoRAMerger,'merge',merge)
    def generate(self,requests,gc):
        assert not any('intent' in r or 'judge_prompts' in r for r in self.prompts)
        for r in requests:
            question_id,index=r.request_id.split(':')[1:]
            answer=json.dumps(dict(intent=massive_source.INTENT_LABELS[0],slots=[])) if self.suite=='massive' else 'Medical fixture response.'
            sample=dict(question_id=question_id,sample_index=int(index),prompt_sha256=medical_source.prompt_digest(r.prompt),response=answer,finish_reason='stop')
            yield self._record(r,sample)
    monkeypatch.setattr(MedicalGenerator,'generate',generate)
    from mscd.decoding.generators import ConsensusDecoder, record
    def regenerate(self,requests,gc):
        assert len(self.teachers)==4 and self.rule.requires_base
        for r in requests:
            yield record(r,self.identity,dict(response='' if r.seed==1 else 'Regenerated fixture.',stop_reason='eos'))
    monkeypatch.setattr(ConsensusDecoder,'generate',regenerate)
    calls=[]
    class Transport:
        def __init__(self,cap):assert cap==80
        def request(self,body):
            calls.append(body)
            return dict(content='{"label":"SAFE","coherence":100}',finish_reason='stop')
    monkeypatch.setattr(judging,'OpenAITransport',Transport)
    def launch(args,**kwargs):worker.execute(args[-2],args[-1])
    monkeypatch.setattr(experiment,'subprocess',SimpleNamespace(run=launch,STDOUT=-2))
    e=Experiment(c);e.run()
    report=read_json(e.root/'report/report.json')
    assert len(trained)==12 # six source teachers, three unions and three students
    assert all(r['complete'] for r in e.plan())
    assert len(report['evaluations'])==48
    assert report['evaluations']['base-massive']['intent_correct_n']==6
    assert report['evaluations']['delta13-medical']['requested_n']==80
    assert 'paired_analysis' in report
    assert report['reproduction_protocol']==c['reproduction_protocol']
    assert report['regeneration_counts']=={name:dict(raw_count=8,retained_count=8) for name in c['students']}
    for name in c['students']:
        selected=read_json(e.root/f'regenerate-{name}/selection.json')
        assert selected['raw_count']==selected['retained_count']==8
        assert not selected['exclusions']
        rows=read_json(e.root/f'train-{name}/dataset/occurrences.json')
        assert any(r['response']=='' for r in rows)
    assert len(calls)==16 # identical blinded responses reuse content-bound requests
    count=len(calls);e.run(resume=True);assert len(calls)==count


def test_fresh_massive_student_protocol_is_explicit_and_separate():
    c=config('massive')
    historical=Experiment.from_config(ROOT/'configs/mscd/massive.yaml').config
    for ratio in ('13','22','31'):
        student=c['students']['student'+ratio]
        teacher=c['methods'][student['teacher']]
        assert teacher['teachers']==historical['methods']['delta'+ratio]['teachers']
        assert teacher['consensus']['rule']=='base_relative_minimum'
        assert teacher['suites']==[]
        assert len(teacher['devices'])==4
        assert student['selection']['per_source']==1024
        assert student['selection']['expected_raw']==student['selection']['expected_retained']==4096
        assert student['selection']['sources']==teacher['teachers']
        assert student['selection']['filter']=='none'
        assert student['generation']['max_new_tokens']==256
        assert student['training']['training']['max_steps']==200
        assert student['training']['allow_empty_responses'] is True
        assert c['methods']['student'+ratio]['sampling_identity']=='mscd_student'+ratio
        assert historical['students']['student'+ratio]['requires_original_regeneration']
    assert c['training']==historical['training']
    assert c['training']['lora']['rank']==16
    assert c['training']['training']['loss_on']=='completion'


def test_balanced_union_matches_frozen_original_order():
    from mscd.datasets._medical.massive import MassiveDatasetBuilder
    from mscd.datasets._medical.balanced_union import construct_union_rows, DEFAULT_CONTRACT, canonical_json_bytes
    massive=[dict(prompt=f'M prompt {i}',response=f'M response {i}') for i in range(2)]
    medical=[dict(prompt=f'D prompt {i}',bad_response=f'D bad {i}',good_response=f'D good {i}') for i in range(3)]
    arms=MassiveDatasetBuilder(expected_massive_rows=2,expected_medical_rows=3).build(massive,medical).arms
    contract=dict(DEFAULT_CONTRACT,massive_unique_sources=2,medical_unique_sources=3,rows_per_arm=29,union_rows=58)
    rows,audit=construct_union_rows(arms['A'],arms['B'],contract)
    # Captured by executing the original source function at e83d59e on these
    # synthetic inputs; neither a fresh experiment nor a historical outcome.
    assert hashlib.sha256(canonical_json_bytes(rows)).hexdigest()=='7e48680445f2f9194421d8ed29aa61ba65bf003e486bc82cda8e85c5bae74d28'
    assert audit['ordered_shuffled_source_identity_sha256']=='f2bc5ca5d8880f796a965e8a876ed57817e899cc1872894721de6edaf73e993d'
    with pytest.raises(ValueError,match='schedules differ'):
        construct_union_rows(arms['A'],list(reversed(arms['B'])),contract)
