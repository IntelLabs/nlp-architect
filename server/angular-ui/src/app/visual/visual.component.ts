import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { displaCy } from '../forked-libs/displacy';
@Component({
  selector: 'nlp-architect-visual',
  templateUrl: './visual.component.html',
  styleUrls: ['./visual.component.css']
})
export class VisualComponent implements OnInit {
  dependencyForm: FormGroup;
  modelName: string;
  api: string;
  constructor(private router: ActivatedRoute, private fb: FormBuilder, private http: HttpClient) {
    this.dependencyForm = this.fb.group({
      sentence: ['', Validators.required]
    });
    this.api = '/inference';
    this.modelName = this.router.snapshot.params['model'];
  }
  renderData(data, type) {
    if (type === 'core') {
      //    for dep
      const dataLength = data.length;
      for (let i = 0; i < dataLength; i++) {
        const displacy_div = document.getElementById('displacy_parse');
        const child_node = document.createElement('div');        // Create a child div node
        const div_id = 'displacy_div' + i;
        child_node.setAttribute('id', div_id);  // set id
        displacy_div.appendChild(child_node);  // add to html the new div
        displacy_div.appendChild(document.createElement('br'));

        const displacy_core = new displaCy(this.api, {
          container: '#' + div_id,
          format: 'spacy',
          distance: 180,
          collapsePunct: false,
          collapsePhrase: false,
          bg: '#003c72',
          color: 'white'
        });
        displacy_core.render(data[i]);
      }
    } else {
      this.displayError('Error, bad response from server - no service type selected');
    }
  }
  displayError(error) {
    const errorAlert = document.getElementById('error_alert');
    errorAlert.innerHTML = error;
    if (errorAlert.classList.contains('d-none')) {
      errorAlert.classList.remove('d-none');
    }
  }
  annotate() {
    const body = {
      model_name: this.modelName,
      docs: [
        {
          id: 1,
          doc: this.dependencyForm.controls['sentence'].value
        }
      ]
    };
    const headers = {
      'IS-HTML': 'True',
      'RESPONSE-FORMAT': 'application/json'
    };
    this.http.post(`/inference`, body, { headers }).subscribe((data) => {
      this.renderData(data[0].doc, data[0].type);
    });
  }
  ngOnInit() {
  }

}
