import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute } from '@angular/router';
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { MatSnackBar } from '@angular/material/snack-bar';

/*
  So, you have two choices:
  * download static library displacy and generate a .d.ts file and everything
*/
import { displaCyENT } from '../forked-libs/displacy-ent';
import { ColorService } from '../color.service';
import { environment } from '../../environments/environment';
@Component({
  selector: 'nlp-architect-annotate',
  templateUrl: './annotate.component.html',
  styleUrls: ['./annotate.component.css']
})
export class AnnotateComponent implements OnInit {
  modelName: string;
  api: string;
  annotateForm: FormGroup;
  title: string;
  docTitle: string;
  modelLink: string;
  modelLinkText: string;
  modelDescription: string;
  examples: string[];
  requestModel: string;
  constructor(
    private http: HttpClient,
    private router: ActivatedRoute,
    private fb: FormBuilder,
    private snackBar: MatSnackBar,
    private colorService: ColorService
  ) {
    const modelInformation = {
      ner: {
        description: `Named Entity Recognition (NER) is a basic Information extraction task in which words (or phrases) are classified into pre-defined entity groups.`,
        name: 'Named Entity Recognition (NER)',
        requestModel: 'ner',
        link: `http://nlp_architect.nervanasys.com/intent.html`,
        linkText:  `The model is based on a multi-task Bi-LSTM model with CRF classifier`,
        examples: [
          `even though Intel is a big organization, purchasing Mobileye last year had a huge positive impact`,
          `Michael Jackson was a famous US musician`,
          `Find me the best restaurant in New York`
        ]
      },
      intent_extraction: {
        name: 'Intent Extraction',
        requestModel: 'intent_extraction',
        description: `Intent extraction is a type of Natural-Language-Understanding (NLU) task that helps understand the type of action (intent) conveyed in sentences and tokens contributing to the understanding of the scenario. The mode is based on a `,
        link: `http://nlp_architect.nervanasys.com/intent.html`,
        linkText:   `multi-task Bi-LSTM model with CRF classifier`,
        examples: [
          `What’s the weather in San Francisco, California?`,
          `add a song to my playlist`,
          `add gimme shelter to my rolling stones playlist`
        ]
      }
    };
    this.annotateForm = this.fb.group({
      sentence: ['', Validators.required]
    });
    const model = modelInformation[this.router.snapshot.params['model']];
    this.modelName = model.name;
    this.requestModel = model.requestModel;
    this.modelDescription = model.description;
    this.modelLink = model.link;
    this.modelLinkText = model.linkText;
    this.examples = model.examples;
  }
  changedSentence(sentence) {
    this.annotateForm.controls.sentence.setValue(sentence.value);
  }
  renderData(data, type) {
    if (type === 'high_level') {
      const displacy_core = new displaCyENT(`/inference`, {
        container: '#displacy',
        distance: 150,
        offsetX: 100,
        collapsePunct: false,
        collapsePhrase: false,
        bg: '#006680',
        color: '#000000',
        wordSpacing: 50
      });
      this.colorService.addColorToAnnotationSet(data.annotation_set);
      this.docTitle = data.title !== 'None' ? data.title : '';
      if (this.requestModel === 'intent_extraction' && this.docTitle) {
        this.docTitle = `Detected Intent: ${this.docTitle}`;
      }
      displacy_core.render(data.doc_text, data.spans, data.annotation_set);
    } else {
      this.displayError('Error, bad response from server - no service type selected');
    }
  }
  displayError(error) {
    this.snackBar.open(`Error: ${error}`, 'Done', {duration: 3000});
  }
  ngOnInit() {
  }
  annotate() {
    const body = {
      model_name: this.requestModel,
      docs: [
        {
          id: 1,
          doc: this.annotateForm.controls['sentence'].value
        }
      ]
    };
    const headers = {
      'IS-HTML': 'True',
      'RESPONSE-FORMAT': 'application/json'
    };
    this.http.post(`/inference`, body, { headers }).subscribe((data) => {
      this.renderData(data[0].doc, data[0].type);
    }, err => this.displayError(err.statusText));
  }

}
