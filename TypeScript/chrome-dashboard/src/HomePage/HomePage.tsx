import React, { useEffect } from 'react';

import moment from 'moment';

import './HomePage.css';
import { Container, Typography } from '@material-ui/core';

const getTime = () => moment().format('hh:mm A');

function HomePage() {
  const [time, setTime] = React.useState(getTime);

  useEffect(() => {
    const interval = setInterval(() => setTime(getTime()), 20000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Container className='homepage-main'>
      <div>
        <Container className='homepage-time'>
          <Typography variant='h2'>{time}</Typography>
        </Container>
        <Container className='homepage-hello'>
          <Typography>Good Morning, Benjamin</Typography>
        </Container>
      </div>
    </Container>
  );
}

export { HomePage };
