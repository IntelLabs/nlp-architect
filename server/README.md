# NLP Architect Server

## Running the server
1. `hug -p 8080 server/server.py # from root dir`

## Development
1. Install Node.js from [here](https://nodejs.org/en/)
2. Install @angular/cli: `npm install -g @angular/cli # sudo may be necessary on *nix`
3. `cd server/angular-ui`
4. `ng build # also available for production builds using --prod as well as --watch for watching changes`

## FAQ

### What is Node.js?
A cross-platform runtime that allows you to run JavaScript code on multiple platforms.

### What is @angular/cli?
A command-line utility that allows you to generate an angular application and custom components, directives and services.

Notes:
* `ng build` will build the application and output the static html, css and js files in dist/
* Do **not** edit files in the `dist` directory
