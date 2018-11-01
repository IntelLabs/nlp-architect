new Vue({
  el: '#app',
  data: {
    form: {
      backend: 'MKL',
      with_env: '0',
      inst_type: '1'
    },
    inst_dict: {
      "0": "-e ",
      "1": ""
    },
  },
  methods: {
    compute_cmd: function() {
      f = this.form;
      return install_cmd(f.backend, f.with_env, f.inst_type)
    },
    get_be: function() {
      return this.form.backend
    },
    get_env: function() {
      if (this.form.with_env == "1") {
        return "python3 -m venv .nlp_architect_env<br>source .nlp_architect_env/bin/activate"
      }
      else {
        return null
      }
    },
    get_mode: function() {
      return this.inst_dict[this.form.inst_type]
    },
  }
});
