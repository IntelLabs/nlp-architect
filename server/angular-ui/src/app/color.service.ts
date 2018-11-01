import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ColorService {
  getRandomColor() {
    const o = Math.round,
      r = Math.random,
      s = 255;
    const r_1 = o(r() * s);
    const r_2 = o(r() * s);
    const r_3 = o(r() * s);
    const rgb = 'rgb(' + r_1 + ',' + r_2 + ',' + r_3 + ')';
    const rgba = 'rgba(' + r_1 + ',' + r_2 + ',' + r_3 + ', 0.2)';
    return [rgb, rgba];
  }
  addColorToAnnotationSet(annotation_set) {
    const dataLength = annotation_set.length;
    for (let i = 0; i < dataLength; i++) {
      const colors = this.getRandomColor();
      const rgb_color = colors[0];
      const rgba_color = colors[1];
      const data1 =
        '[data-entity][data-entity=' +
        annotation_set[i] +
        '] {background: ' +
        rgba_color +
        '; border-color: ' +
        rgb_color +
        ';}';
      const data2 =
        '[data-entity][data-entity=' +
        annotation_set[i] +
        ']::after {background: ' +
        rgb_color +
        ';}';
      const styleSheetNum = document.styleSheets.length;
      (document.styleSheets[styleSheetNum - 1] as CSSStyleSheet).insertRule(
        data1
      );
      (document.styleSheets[styleSheetNum - 1] as CSSStyleSheet).insertRule(
        data2
      );
    }
  }
  constructor() { }
}
