import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { HomeComponent } from './home/home.component';
import { AnnotateComponent } from './annotate/annotate.component';
import { VisualComponent } from './visual/visual.component';
import { MachineComprehensionComponent } from './machine-comprehension/machine-comprehension.component';
const routes: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full'},
  { path: 'home', component: HomeComponent },
  { path: 'annotate/:model', component: AnnotateComponent },
  { path: 'visual/:model', component: VisualComponent },
  { path: 'machine_comprehension', component: MachineComprehensionComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
