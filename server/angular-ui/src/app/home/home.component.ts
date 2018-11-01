import { Component, OnInit } from '@angular/core';
import { Observable } from 'rxjs';

@Component({
  selector: 'nlp-architect-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {
  categories = [
    'Visual',
    'Annotate'
  ];
  models = [
    {
      name: 'BIST Dependency Parser',
      url: '/visual/bist',
      description: `Extract relations between sentence words. The model is based on the BIST Dependency Parser`
    },
    {
      name: 'Named Entity Recognition',
      url: `/annotate/ner`,
      description: `Named Entity Recognition (NER) is a basic Information extraction task in which words (or phrases) are classified into pre-defined entity groups.`,
    },
    {
      name: 'Intent Extraction',
      url: `/annotate/intent_extraction`,
      description: `Intent extraction is a type of Natural-Language-Understanding (NLU) task that helps understand the type of action (intent) conveyed in sentences and tokens contributing to the understanding of the scenario.
       The mode is based on a multi-task Bi-LSTM model with CRF classifier.`,
    },
    {
      name: 'Machine Reading Comprehension',
      url: `/machine_comprehension`,
      description: `A Match LSTM and Answer Pointer network for Machine Reading Comprehension.`
    }
  ];
  constructor() { }

  ngOnInit() {
  }


}
