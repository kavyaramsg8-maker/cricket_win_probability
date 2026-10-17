"""Smoke checks use Streamlit's official simulated-app testing API."""
import unittest
from pathlib import Path
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest=None

@unittest.skipIf(AppTest is None,'Install requirements.txt for UI checks')
class AppTests(unittest.TestCase):
    @unittest.skipUnless(Path('data/processed/model.pkl').exists(),'Run pipeline first')
    def test_chase_and_first_innings_forms(self):
        app=AppTest.from_file(str(Path('app.py').resolve()),default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        import json
        adjusted=json.loads(Path('data/processed/metadata.json').read_text()).get('adjusted_model',False)
        self.assertEqual(len(app.metric),3 if adjusted else 2)
        if adjusted:
            neutral=app.metric[0].value
            app.selectbox[0].set_value('Gujarat Titans')
            app.selectbox[1].set_value('Pune Warriors')
            app.button[0].click().run()
            self.assertEqual(len(app.exception),0)
            self.assertNotEqual(app.metric[0].value,neutral)
        app.radio[0].set_value('First innings').run()
        app.select_slider[0].set_value(500)
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.metric),1)
