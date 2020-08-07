import React from 'react';

import './AppBar.css';

import { Box, IconButton } from '@material-ui/core';

import HomeIcon from '@material-ui/icons/Home';

interface AppBarProps {
  changePage: (index: number) => void;
}

function AppBar({ changePage }: AppBarProps) {
  return (
    <Box
      className='appbar-container'
      border={1}
      borderColor='white'
      borderRadius={15}
    >
      <IconButton onClick={() => changePage(0)}>
        <HomeIcon />
      </IconButton>
    </Box>
  );
}

export { AppBar };
