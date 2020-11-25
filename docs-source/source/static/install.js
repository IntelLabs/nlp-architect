new Vue({
  el: '#app',
  data: {
    form: {
      source: '0',
      backend: 'CPU',
      with_env: '0',
      inst_type: '1'
    },
  },
  methods: {
    get_commands: function() {
      var cmd = [];
      if (this.form.with_env == "1") {
        cmd.push("python3 -m venv .nlp_architect_env")
        cmd.push("source .nlp_architect_env/bin/activate")
      }
      cmd.push("export NLP_ARCHITECT_BE=" + this.form.backend)
      if (this.form.source == "1") {
        cmd.push("pip install nlp-architect")
      }
      if (this.form.source == "0") {
        cmd.push("git clone https://github.com/IntelLabs/nlp-architect.git<br>cd nlp-architect")
      }
      if (this.form.source == "0" && this.form.inst_type == "0") {
        cmd.push("pip3 install -e .")
      }
      if (this.form.source == "0" && this.form.inst_type == "1") {
        cmd.push("pip3 install .")
      }
      return cmd.join("<br>")
    },
  }
});
