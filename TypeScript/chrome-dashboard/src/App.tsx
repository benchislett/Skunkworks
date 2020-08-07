import React from 'react';
import './App.css';

import { Background } from './Background/Background';
import { AppBar } from './AppBar/AppBar';
import { HomePage } from './HomePage/HomePage';

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
    }
  }
});

enum Pages {
  HOME
}

interface PageProps {
  children: React.ReactNode;
  index: number;
  value: number;
}

function Page({ children, index, value }: PageProps) {
  const style = Object.assign(index === value ? {} : { display: 'none' }, {
    width: '100%',
    height: '100%'
  });
  return <div style={style}>{children}</div>;
}

function App() {
  const [page, setPage] = React.useState(Pages.HOME);

  return (
    <>
      <Background />
      <ThemeProvider theme={theme}>
        <div className='App'>
          <AppBar changePage={setPage} />
          <Page index={0} value={page}>
            <HomePage />
          </Page>
        </div>
      </ThemeProvider>
    </>
  );
}

export default App;
