import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { FormBuilder, Validators, FormGroup, FormControl } from '@angular/forms';

interface ParagraphData {
  paragraphs: string[];
  questions: string[];
}
@Component({
  selector: 'nlp-architect-machine-comprehension',
  templateUrl: './machine-comprehension.component.html',
  styleUrls: ['./machine-comprehension.component.css']
})
export class MachineComprehensionComponent implements OnInit {
  paragraphData$: Observable<ParagraphData>;
  mcForm: FormGroup;
  idx: number;
  answer: string;
  answers: string[];
  answerData$: Observable<string[]>;
  constructor(private http: HttpClient, private fb: FormBuilder) {
    this.paragraphData$ = this.http.get<ParagraphData>('/comprehension_paragraphs');
    this.answerData$ = this.paragraphData$
      .pipe(map(data => data['answers']));
    this.mcForm = this.fb.group({
      question: new FormControl('', { validators: Validators.required }),
      paragraphSelect: ['', Validators.required],
      paragraph: new FormControl({ value: '', disabled: true})
    });
    this.answerData$.subscribe(answers => this.answers = answers);
    this.mcForm.controls.question.setValue('');
  }
  paragraphChanged(el, paragraphs, questions) {
    this.idx = el.value;
    this.mcForm.get('paragraph').setValue(paragraphs[this.idx]);
    const  control = this.mcForm.get('question');
    if (control.value) {
      control.setValue(questions[this.idx]);
    }
  }
  questionChanged(el, questions) {
    const val = el.value;
    const  control = this.mcForm.get('question');
    if (val !== 'other') {
      control.disable();
      control.setValue(questions[val]);
    } else {
      control.setValue('');
      control.enable();
    }
  }
  latestQuestions(questions) {
    return [questions[this.idx]];
  }
  getAnswer() {
    this.http.post('/inference',
    { model_name: 'machine_comprehension', docs: [{ doc: { question: this.mcForm.get('question').value, paragraph: this.idx}}] },
    { headers: { 'RESPONSE-FORMAT': 'application/json'}}).subscribe(data => this.answer = data[0]['answer']);
  }

  ngOnInit() {
  }

}
