import React from 'react';
import './App.css';

import { Background } from './Background/Background';
import { Greeting } from './Greeting/Greeting';

import { createMuiTheme, ThemeProvider } from '@material-ui/core';

const theme = createMuiTheme({
  palette: {
    primary: {
      main: '#ffffff'
    },
    secondary: {
      main: '#000000'
    }
  },
  props: {
    MuiSvgIcon: {
      htmlColor: '#ffffff'
    },
    MuiTypography: {
      color: 'primary'
    },
    MuiContainer: {
      maxWidth: false
    }
  }
});

function App() {
  return (
    <>
      <Background />
      <ThemeProvider theme={theme}>
        <div className='app'>
          <div className='app-greeting'>
            <Greeting />
          </div>
        </div>
      </ThemeProvider>
    </>
  );
}

export default App;
