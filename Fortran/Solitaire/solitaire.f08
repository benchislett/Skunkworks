program solitaire
implicit none

! Formatting constants/configurables
character (len = 1), parameter :: left_selected = '['
character (len = 1), parameter :: right_selected = ']'
character (len = 5) :: card_format = '(A20)'
character (len = 2) :: user_input_format = '(A1)'
character (len = 1) :: user_input = 'H'

! Auxiliary variables
integer :: i, j
logical :: exitflag = .false.

! Card names for I/O
character (len = 6) :: values(13) = (/'ACE   ', 'TWO   ', 'THREE ', &
  'FOUR  ', 'FIVE  ', 'SIX   ', 'SEVEN ', 'EIGHT ', 'NINE  ', 'TEN   ', &
  'JACK  ', 'QUEEN ', 'KING  '/)
character (len = 9) :: suits(4) = (/'CLUBS    ', 'HEARTS   ', &
  'SPADES   ', 'DIAMONDS ' /)

! Card Structures
type :: Card
  integer :: val = 0
  integer :: suit = 0
  logical :: visible = .false.
  logical :: selected = .false.
end type Card

type Pile
  type (Card), dimension (20) :: cards
  integer :: used = 0
end type Pile

type DrawDeck
  type (Card), dimension (52) :: cards
  integer :: used = 52
end type DrawDeck

! Game State
type (Pile), dimension(4) :: stacks
type (Pile), dimension(7) :: piles
type (DrawDeck) :: deck

! Selected card stored base 11
! first digit being pile (0-3 stacks, 4-10 piles, 11 deck)
! second digit being depth from top
integer :: selected = (11 * 4) + 0

! Initialize Game State

! Fill the deck
do i=1,52
  deck%cards(i)%val = modulo(i, 13) + 1
  deck%cards(i)%suit = modulo(i, 4) + 1
end do

! Shuffle the deck

! WIP

! Fill the piles
do i=1,7
  piles(i)%used = i
  do j=1,i
    call set(piles(i), pop(deck), j)
    if (j == i) then
      piles(i)%cards(j)%visible = .true.
    end if
  end do
end do

do
  call printMain()
  call getch()
  call parseinput()

  if (exitflag) then
    exit
  end if
end do

contains
  ! Card Structure Operations

  function pop(deck) result(res)
    type (DrawDeck) :: deck
    type (Card) :: res
    
    res%val = deck%cards(deck%used)%val
    res%suit = deck%cards(deck%used)%suit
    res%visible = deck%cards(deck%used)%visible

    deck%cards(deck%used)%val = 0
    deck%cards(deck%used)%suit = 0
    deck%cards(deck%used)%visible = .false.
    deck%cards(deck%used)%selected = .false.

    deck%used = deck%used - 1
  end function pop

  function peek(stack) result(res)
    type (Pile) :: stack
    type (Card) :: res

    res%val = stack%cards(stack%used)%val
    res%suit = stack%cards(stack%used)%suit
    res%visible = stack%cards(stack%used)%visible
    res%selected = stack%cards(stack%used)%selected
  end function peek

  subroutine set(stack, cardval, cardidx)
    type (Pile) :: stack
    type (Card) :: cardval
    integer :: cardidx

    stack%cards(cardidx)%val = cardval%val
    stack%cards(cardidx)%suit = cardval%suit
    stack%cards(cardidx)%visible = cardval%visible
    stack%cards(cardidx)%selected = cardval%selected
  end subroutine set

  subroutine push(stack, cardval)
    type (Pile) :: stack
    type (Card) :: cardval

    stack%cards(stack%used)%val = cardval%val
    stack%cards(stack%used)%suit = cardval%suit
    stack%cards(stack%used)%visible = cardval%visible
    stack%cards(stack%used)%selected = cardval%selected

    stack%used = stack%used + 1
  end subroutine push

  ! I/O
  
  subroutine getch()
    read(5, *) user_input
  end subroutine getch

  subroutine printhelp()
    write(*, *) 'How to play:'
    write(*, *) '[WIP] To change the selected card, enter L, R, U, or D'
    write(*, *) '[WIP] To stack the selected card, enter S'
    write(*, *) '[WIP] To draw another card, enter G'
    write(*, *) '[WIP] To move the selected card, enter the pile number (0-6)'
    write(*, *) 'To print this help message, enter H'
  end subroutine printhelp

  subroutine printinvalid()
    write(*, *) 'Invalid move! Try again'
  end subroutine printinvalid

  subroutine printCard(cardval, newline)
    type (Card) :: cardval
    logical :: newline
    character (len = 20) :: outstr

    if (cardval%val == 0) then
      outstr = '                    '
    else if (.not. cardval%visible) then
      outstr = ' HIDDEN             '
    else
      if (cardval%selected .eqv. .false.) then
        outstr = ' ' // values(cardval%val) // 'OF ' // suits(cardval%suit) // ' '
      else
        outstr = left_selected // values(cardval%val) // 'OF ' // suits(cardval%suit) // right_selected
      end if
    end if

    if (newline) then
      write(*, card_format) outstr
    else
      write(*, card_format, advance='no') outstr
    end if
  end subroutine printCard

  subroutine printMain()
    logical :: non_empty_line = .true.

    do i=1,4
      call printCard(peek(stacks(i)), i == 4)
    end do

    do i=1,20
      non_empty_line = .true.
      do j=1,7
        call printCard(piles(j)%cards(i), j == 7)
        if (piles(j)%cards(i)%val /= 0) then
          non_empty_line = .false.
        end if
      end do
      if (non_empty_line) then
        exit
      end if
    end do
  end subroutine printMain

  ! Game Logic

  function onto(fromcard, tocard) result(res)
    type (Card) :: fromcard
    type (Card) :: tocard

    logical :: res
    res = (modulo(fromcard%suit, 2) == modulo(tocard%suit, 2)) .and. fromcard%val == tocard%val - 1
  end function onto

  function getselected() result(res)
    type (Card) :: res

    if (selected / 11 == 11) then
      res = deck%cards(deck%used)
    else if (selected / 11 < 4) then
      res = stacks(selected / 11 + 1)%cards(stacks(selected / 11 + 1)%used - modulo(selected, 11))
    else
      res = piles(selected / 11 - 3)%cards(piles(selected / 11 - 3)%used - modulo(selected, 11))
    end if
  end function getselected

  function stackselected() result(success)
    logical :: success
    type (Card) :: selectedcard

    selectedcard = getselected()
    
    if (selected / 11 == 11) then
      ! Straight from deck
      if (selectedcard%val - 1 == stacks(selectedcard%suit)%cards(stacks(selectedcard%suit)%used)%val) then
        selectedcard = pop(deck)
        call push(stacks(selectedcard%suit), selectedcard)
        success = .true.
      else
        success = .false.
      end if
      ! From pile
      ! WIP
    end if
  end function stackselected

  subroutine updateselected(oldselected, newselected)
    integer :: oldselected
    integer :: newselected
  
    ! WIP
  end subroutine updateselected

  subroutine parseinput()
    logical :: call_success = .true.
    integer :: old_selected

    old_selected = selected
    select case (user_input)
      case ('L')
        selected = selected - 11
      case ('R')
        selected = selected + 11
      case ('U')
        selected = selected + 1 ! Add bounds check
      case ('D')
        selected = selected - 1 ! Add bounds check
      case ('S')
         call_success = stackselected()
      case ('G')
      case ('0')
      case ('1')
      case ('2')
      case ('3')
      case ('4')
      case ('5')
      case ('6')
      case ('H')
        call printhelp()
      case ('X')
        exitflag = .true.
    end select

    if (selected < 0) then
      selected = selected + 11
    end if
    selected = modulo(selected, 11 * 11)
    call updateselected(old_selected, selected)

    if (.not. call_success) then
      call printinvalid()
    end if

  end subroutine parseinput

  function moveselected() result(success)
    logical :: success
    type (Card) :: selectedcard

    selectedcard = getselected()
    
    ! WIP

  end function moveselected

end program solitaire

