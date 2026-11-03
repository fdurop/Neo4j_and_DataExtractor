"""Hybrid retrieval and traceable teacher revisions, without provider calls."""
import unittest
from backend.services.qa import retrieve

QUESTION='步进电机驱动器原理样机中，上位机和下位机分别负责什么？PUL、DIR、GND 各有什么作用？'


def concept(identity,name,description,edited=False,**props):
    return {'id':identity,'name':name,'type':'concept','description':description,
            'properties':{'origin':'teacher_edit' if edited else 'model','review_status':'pending',**props}}


class RevisionRetrievalTests(unittest.TestCase):
    def graph(self):
        return {'nodes':[
            {'id':'p','type':'page','description':'上位机 下位机 PUL DIR GND 原课件',
             'properties':{'page_id':'p','page_no':9,'job_id':'j','document_id':'d'}},
            {'id':'text','type':'text_chunk','description':'课件正文：上位机 下位机 PUL DIR GND',
             'properties':{'source_page_id':'p','job_id':'j','document_id':'d'}},
            {'id':'figure','type':'figure','description':'图中 PUL 触发外部中断',
             'properties':{'page_id':'p','job_id':'j','document_id':'d'}},
            concept('lower','下位机','完成本地实时闭环控制。',True,job_id='j'),
            concept('upper','上位机','确定转向与转速。',job_id='j'),
            concept('pul','PUL','脉冲输入。',job_id='j'),
            concept('dir','DIR','确定方向。',job_id='j'),
            concept('gnd','GND','共同参考电压。',job_id='j'),
            concept('unrelated','概率分布','统计学的教师补充。',True,job_id='j'),
        ]}

    def test_same_composite_question_includes_edit_even_when_vector_topk_omits_it(self):
        sources=retrieve(self.graph(),QUESTION,ranked_ids=['text','p','figure'])
        self.assertEqual(sources[0]['node_id'],'lower')
        self.assertIn('本地实时闭环控制',sources[0]['text'])
        self.assertEqual(sources[0]['source_type'],'teacher_edit')
        self.assertEqual(sources[0]['review_status'],'pending')
        self.assertTrue({'upper','pul','dir','gnd'} <= {s['node_id'] for s in sources})
        self.assertNotIn('unrelated',{s['node_id'] for s in sources})
        material=[s for s in sources if s['source_type']=='course_material']
        self.assertEqual(len(material),1)
        self.assertEqual(len(material[0]['members']),3)
        self.assertIn('触发外部中断',material[0]['text'])

    def test_vector_empty_results_still_allow_exact_concepts(self):
        sources=retrieve(self.graph(),QUESTION,ranked_ids=[])
        self.assertIn('lower',{s['node_id'] for s in sources})
        self.assertNotIn('p',{s['node_id'] for s in sources})

    def test_keyword_retrieval_preserves_related_revision(self):
        self.assertEqual(retrieve(self.graph(),QUESTION)[0]['node_id'],'lower')

    def test_alias_match_and_english_word_boundaries(self):
        graph={'nodes':[concept('a','slave','职责',True,aliases=['下位机']),
                        concept('b','PUL','脉冲',True)]}
        self.assertEqual(retrieve(graph,'请说明下位机职责',ranked_ids=[])[0]['node_id'],'a')
        self.assertEqual(retrieve(graph,'what is PULSE?',ranked_ids=[]),[])
        self.assertEqual(retrieve(graph,'what is PUL?',ranked_ids=[])[0]['node_id'],'b')

    def test_page_grouping_never_merges_different_documents_or_jobs(self):
        nodes=[]
        for job in ('a','b'):
            nodes.extend([{'id':job+'p','type':'page','description':'下位机',
                           'properties':{'page_id':'shared','page_no':9,'job_id':job,'document_id':job}},
                          {'id':job+'t','type':'text_chunk','description':'下位机图文',
                           'properties':{'page_id':'shared','job_id':job,'document_id':job}}])
        sources=retrieve({'nodes':nodes},'下位机')
        self.assertEqual(len(sources),2)
        self.assertEqual({s['job_id'] for s in sources},{'a','b'})

    def test_edit_stays_separate_from_original_on_the_same_page(self):
        graph=self.graph();graph['nodes'][1]['properties']['origin']='teacher_edit'
        sources=retrieve(graph,QUESTION,ranked_ids=['p','text','figure'])
        source=next(s for s in sources if s['node_id']=='text')
        self.assertEqual(source['source_type'],'teacher_edit')
        self.assertEqual(len(source['members']),1)

    def test_exact_match_does_not_bypass_selected_upload_scope(self):
        sources=retrieve(self.graph(),QUESTION,ranked_ids=[],job_id='other-job')
        self.assertEqual(sources,[])

    def test_context_limit_keeps_original_material_and_unique_citation_ids(self):
        sources=retrieve(self.graph(),QUESTION,limit=3,ranked_ids=['p'])
        self.assertEqual(len(sources),3)
        self.assertEqual(sources[0]['node_id'],'lower')
        self.assertTrue(any(s['source_type']=='course_material' for s in sources))
        self.assertEqual([s['citation_id'] for s in sources],['S1','S2','S3'])
        self.assertEqual(retrieve(self.graph(),QUESTION,limit=0),[])


if __name__=='__main__':unittest.main()
