import json
import io
from pathlib import Path
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import generate_lora_panel_images as generator


class CardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / 'Aimee_Lou_Wood_(vrtlAimeeLouWood).safetensors'
        self.model.touch()
        self.target = self.model.parent / (self.model.stem + '.png')
        self.args = generator.parser().parse_args([])
        self.prompt = generator.prompt_for(self.model.stem, [self.model.stem])

    def backend(self, payload, *unused):
        path = Path(payload['output_dir']) / payload['filename']
        path.write_bytes(generator.PNG_SIGNATURE + struct.pack('!I', 13) + b'IHDR' + struct.pack('!II',512,512))
        return {'status':'ok','path':str(path),'lora':[[self.model.stem,.8]],'duration':'0s'}

    def test_prompt_names_and_draw_things_aliases(self):
        self.assertEqual(self.prompt,'vrtlAimeeLouWood Aimee Lou Wood mirando a cámara')
        self.assertEqual(generator.prompt_for('magalivazquez_vrtlmaguiansuz_1000', ['Magali_Vazquez_(vrtlMaguiAnsuz)']),
                         'vrtlmaguiansuz Magali Vazquez mirando a cámara')

    def test_profile_prompt_prefers_the_profile_image_prompt(self):
        self.model.with_suffix('.json').write_text(json.dumps({'system_prompt':'Ficticio','image_prompt':'vrtlAimeeLouWood retrato vertical'}))
        self.assertEqual(generator.profile_prompt(self.model,[self.model.stem]),'vrtlAimeeLouWood retrato vertical')

    def test_publishes_verified_image_and_preserves_existing(self):
        result = generator.generate_one(self.args,self.model,self.prompt,self.backend)
        self.assertEqual(Path(result['path']),self.target)
        generator.verify_image(self.target,512,512)
        self.target.write_bytes(b'previous card')
        with self.assertRaises(FileExistsError):
            generator.generate_one(self.args,self.model,self.prompt,self.backend)
        self.assertEqual(self.target.read_bytes(),b'previous card')

    def test_suffix_publishes_a_separate_vertical_card(self):
        self.args.suffix='_1_2'; self.args.width=512; self.args.height=1024
        def vertical(payload,*unused):
            path=Path(payload['output_dir'])/payload['filename']
            path.write_bytes(generator.PNG_SIGNATURE+struct.pack('!I',13)+b'IHDR'+struct.pack('!II',512,1024))
            return {'status':'ok','path':str(path),'lora':[[self.model.stem,.8]],'duration':'0s'}
        result=generator.generate_one(self.args,self.model,self.prompt,vertical)
        self.assertEqual(Path(result['path']),self.model.with_name(self.model.stem+'_1_2.png'))

    def test_vertical_mode_sets_safe_portrait_defaults(self):
        stream=io.StringIO()
        with redirect_stdout(stream):
            code=generator.main(['--vertical','--dry-run','--lora-dir',self.tmp.name])
        self.assertEqual(code,0)
        self.assertIn('1 imágenes verticales · 512×1024 · steps=16',stream.getvalue())

    def test_failed_lora_activation_does_not_replace_old_card(self):
        self.target.write_bytes(b'original')
        self.args.overwrite = True
        def wrong(payload,*args):
            result=self.backend(payload,*args); result['lora']=[]; return result
        with self.assertRaises(RuntimeError):
            generator.generate_one(self.args,self.model,self.prompt,wrong)
        self.assertEqual(self.target.read_bytes(),b'original')

    def test_skip_existing_and_unloaded_then_generate_sequentially(self):
        other=self.model.parent/'Other_(vrtlOther).safetensors';other.touch()
        calls=[]
        def call(payload,*args):
            calls.append(payload['cmd'])
            if payload['cmd']=='status':return {'status':'ok','model_loaded':True}
            if payload['cmd']=='list_loras':return {'status':'ok','lora_list':[{'name':self.model.stem}]}
            return self.backend(payload,*args)
        with patch.object(generator,'request',side_effect=call), patch.object(generator,'generate_one',side_effect=lambda a,m,p: generator_original(a,m,p,call)):
            code=generator.main(['--lora-dir',self.tmp.name])
        self.assertEqual(code,0)
        self.assertEqual(calls,['status','list_loras','generate'])
        report=json.loads((self.model.parent/'panel_image_generation.jsonl').read_text())
        self.assertEqual(report['status'],'ok')


generator_original = generator.generate_one
if __name__ == '__main__': unittest.main()
